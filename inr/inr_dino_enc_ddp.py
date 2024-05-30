import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from coral.losses import batch_mse_rel_fn
from coral.metalearning import outer_step_dino
from coral.mlp import MLP, Derivative, ResNet
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, rearrange
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.plot import show
from coral.utils.init_environ import init_environ 

def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        # self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + 1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, num_channels)
        # self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = x.permute(0, 2, 3, 1)
        x = x.mean(-1).mean(-1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class INRwCODE(nn.Module):
    def __init__(self, inr, n, t, code):
        super().__init__()
        self.inr = inr 
        # self.code = nn.parameter.Parameter(torch.zeros(n*t, code)) # -> (n,t,code)
        self.n = n 
        self.t = t 
        self.code = FNO2d(12, 12, 32, code)

    # def get_code(self, idx):

    # def forward(self, x):
    def modulated_forward(self, x, latent):
        return self.inr.modulated_forward(x, latent)

@hydra.main(config_path="config/", config_name="siren.yaml")
def main(cfg: DictConfig) -> None:

    init_environ(cfg)
    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)

    # submitit.JobEnvironment()
    # data
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        # cfg = checkpoint['cfg']
    elif saved_checkpoint == False:
        #wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name

    #data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    sub_from = cfg.data.sub_from
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    gamma_step = cfg.optim.gamma_step
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    lr_adapt = cfg.optim.lr_adapt
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    n_steps = cfg.optim.n_steps

    update_modulations=cfg.optim.update_modulations
    # print(epochs)
    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim

    # wandb
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"),
                     f"wandb/{cfg.wandb.dir}/{dataset_name}")
        if cfg.wandb.dir is not None
        else None
    )

    sweep_id = cfg.wandb.sweep_id
    device = torch.device("cuda")
    print("run dir given", run_dir)
    if cfg.rank ==0 :
        run = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            id=run_id,
            dir=run_dir,
            resume='allow',
        )
        if run_dir is not None:
            os.symlink(run.dir.split("/files")[0], run_dir)

        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        run_name = wandb.run.name

        print("id", run.id)
        print("dir", run.dir)

        if data_to_encode is not None:
            RESULTS_DIR = (
                Path(os.getenv("WANDB_DIR")) / dataset_name / data_to_encode / "inr"
            )
        else:
            RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"

        os.makedirs(str(RESULTS_DIR), exist_ok=True)
    
    set_seed(seed)

    (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        seq_inter_len=seq_inter_len,
        seq_extra_len=seq_extra_len,
        sub_from=sub_from,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
    )
    print(f"data: {dataset_name}, u_train: {u_train.shape}, u_eval_extrapolation: {u_eval_extrapolation.shape}, u_test: {u_test.shape}")
    u_test = u_test[..., 0:1]
    print(f"use the initial conditions for test, u_test:{u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}")
    print("same_grid : ", same_grid)

    if cfg.rank ==0 :
        if data_to_encode == None:
            run.tags = ("inr",) + (model_type,) + (dataset_name,) + (f"sub={sub_tr}",)
        else:
            run.tags = (
                ("inr",)
                + (model_type,)
                + (dataset_name,)
                + (f"sub={sub_tr}",)
                + (data_to_encode,)
            )

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, latent_dim, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, latent_dim, dataset_name, data_to_encode
    )

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = trainset.input_dim
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = trainset.output_dim

    # transforms datasets shape into (N * T, Dx, Dy, C)
    trainset = rearrange(trainset, dataset_name)
    testset = rearrange(testset, dataset_name)

    #total frames trainset
    ntrain = trainset.z.shape[0]
    #total frames testset
    ntest = testset.z.shape[0]

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        batch_size = int(batch_size / torch.distributed.get_world_size())
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        # val_sampler = None
    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
    ) 

    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    # alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))
    # inr.register_parameter('alpha',alpha)
    inr_w_code = INRwCODE(inr, u_train.shape[0], u_train.shape[-1], latent_dim).to(device)

    if cfg.distributed:
        inr_w_code = torch.nn.parallel.DistributedDataParallel(inr_w_code, device_ids=[cfg.gpu_id])
        inr_without_ddp = inr_w_code.module 
    else:
        inr_without_ddp = inr_w_code

    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code
    
    params_inr = []
    params_code = []
    for name, param in inr_w_code.named_parameters():
        if 'code' in name:
            params_code.append(param)
        else:
            params_inr.append(param)

    update_alpha = cfg.optim.update_alpha

    optimizer = torch.optim.AdamW(
        [
            {"params": params_inr, "lr": lr_inr},
            {"params": params_code, "lr": lr_code},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    # if update_alpha:
    #     optimizer = torch.optim.AdamW(
    #     [
    #         {"params": params_inr, "lr": lr_inr},
    #         {"params": params_alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
    #     ],
    #     lr=lr_inr,
    #     weight_decay=0,
    # )
    # else:
    #     # print('not optmize alpha')
    #     optimizer = torch.optim.AdamW(
    #         [
    #             {"params": params_inr, "lr": lr_inr},
    #             # {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
    #         ],
    #         lr=lr_inr,
    #         weight_decay=0,
    #     )

    if saved_checkpoint:
        inr_without_ddp.load_state_dict(checkpoint['inr'])
        optimizer.load_state_dict(checkpoint['optimizer_inr'])
        epoch_start = checkpoint['epoch']
        # alpha = checkpoint['alpha']
        best_loss = checkpoint['loss']
        # cfg = checkpoint['cfg']
        print("epoch_start, best_loss", epoch_start, best_loss)
        print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        best_loss = np.inf

    if cfg.rank == 0:
        wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma_step,
        patience=500,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )
    rel_train_mse = torch.tensor([0.0]).to(device)
    # rel_test_mse = torch.tensor([0.0]).to(device)
    fit_train_mse = torch.tensor([0.0]).to(device)
    # fit_test_mse = torch.tensor([0.0]).to(device)
    for step in range(epoch_start, epochs):
        rel_train_mse[0] = 0
        rel_test_mse = 0
        fit_train_mse[0] = 0
        fit_test_mse = 0
        use_rel_loss = step % 10 == 0
        step_show = step % 100 == 0
        # use_rel_loss = step % 1 == 0
        # step_show = step % 1 == 0
        step_show_last = step == epochs - 1
        if cfg.distributed:
            train_loader.sampler.set_epoch(step)
        for substep, (images, modulations, coords, idx) in enumerate(train_loader):
            # print(alpha)
            inr_w_code.train()
            images = images.to(device)
            # modulations = modulations.to(device)
            coords = coords.to(device)
            n_samples = images.shape[0]
            # input_modulations = inr_without_ddp.code[idx]
            input_modulations = inr_without_ddp.code(images)
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # print(images.shape)
            # if not update_modulations:
            #     input_modulations = torch.zeros_like(modulations)
            # else:
            #     input_modulations = modulations
            #     raise NotImplementedError
            # import pdb; pdb.set_trace()
            # print(modulations.mean())
            outputs = outer_step_dino(
                inr_w_code,
                coords,
                images,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
                # modulations=torch.zeros_like(modulations),
                modulations=input_modulations,
            )

            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            # nn.utils.clip_grad_value_(inr_w_code.parameters(), clip_value=1.0)
            optimizer.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse[0] += loss.item() * n_samples

            # mlp regression
            if use_rel_loss:
                rel_train_mse[0] += outputs["rel_loss"].item() * n_samples

            # debug: visualize 
            # reconstructions = outputs["reconstructions"]
            # from visualize import write_image_pair 
            # # import pdb; pdb.set_trace()
            # write_image_pair(images.detach().cpu().numpy(), reconstructions.detach().cpu().numpy(), 0, path=os.path.join(run.dir, 'pred.png'))

            # if update_modulations:
            #     trainset[idx] = outputs["modulations"].detach().cpu()
        if cfg.distributed:
            dist.all_reduce(fit_train_mse, op=dist.ReduceOp.SUM, async_op=False)
        train_loss = fit_train_mse / ntrain

        if model_type=="fourier_features":
            scheduler.step(train_loss)

        if use_rel_loss:
            if cfg.distributed:
                dist.all_reduce(rel_train_mse, op=dist.ReduceOp.SUM, async_op=False)
            rel_train_loss = rel_train_mse / ntrain

        if cfg.rank == 0 and (True in (step_show, step_show_last)):
            set_requires_grad(inr_without_ddp, False)
            # save_best = True 
            # assert len(test_loader) == 1
            for images, modulations, coords, idx in test_loader:
                inr_w_code.eval()
                images = images.to(device)
                modulations = modulations.to(device)
                coords = coords.to(device)
                n_samples = images.shape[0]

                states_params_index = inr_without_ddp.code(images)
                outputs = outer_step_dino(
                    inr_w_code,
                    coords,
                    images,
                    is_train=False,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    modulations=states_params_index,
                )

                loss = outputs["loss"]
                fit_test_mse += loss.item() * n_samples

                if use_rel_loss:
                    rel_test_mse += outputs["rel_loss"].item() * n_samples

            test_loss = fit_test_mse / ntest

            if use_rel_loss:
                rel_test_loss = rel_test_mse / ntest
            set_requires_grad(inr_without_ddp, True)

        if cfg.rank == 0:
            if True in (step_show, step_show_last):
                wandb.log(
                    {
                        "test_rel_loss": rel_test_loss,
                        "train_rel_loss": rel_train_loss,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                    },
                )

            else:
                wandb.log(
                    {
                        "train_loss": train_loss,
                    },
                    step=step,
                    commit=not step_show,
                )

        if train_loss < best_loss:
            best_loss = train_loss
            if cfg.rank == 0:
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr": inr_without_ddp.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": best_loss,
                        # "alpha": alpha,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )

    return rel_test_loss

if __name__ == "__main__":
    main()