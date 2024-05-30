from coral.utils.plot import show
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.load_inr import create_inr_instance, load_inr_model
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from coral.utils.data.load_modulations import load_dynamics_modulations
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import (KEY_TO_INDEX, TemporalDatasetWithCode)
# from coral.mlp import Derivative
from torchdiffeq import odeint
from omegaconf import DictConfig, OmegaConf
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import hydra
import einops
import os
import sys
from pathlib import Path
from dynamics_modeling.eval_fno_ode import batch_eval_loop

sys.path.append(str(Path(__file__).parents[1]))

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
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + num_channels, self.width)
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

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        # self.grid = None 
        self.grid_per_bs = None

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape, grid.shape)
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
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        if self.grid_per_bs is not None:
            return self.grid_per_bs.repeat(shape[0],1,1,1)
        else:
            batchsize, size_x, size_y = shape[0], shape[1], shape[2]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat(
                [batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat(
                [batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(device)
            self.grid_per_bs = grid[0:1]
            return grid
        # return torch.cat((gridx, gridy), dim=-1).to(device)

class Derivative(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels, **kwargs):
        super().__init__()
        self.net = FNO2d(modes1, modes2, width, num_channels)

    def forward(self, t, u):
        return self.net(u)
    

class DetailedMSE():
    def __init__(self, keys, dataset_name="shallow-water-dino", mode="train", n_trajectories=256):
        self.keys = keys
        self.mode = mode
        self.dataset_name = dataset_name
        self.n_trajectories = n_trajectories
        self.reset_dic()

    def reset_dic(self):
        dic = {}
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] = 0
        self.dic = dic

    def aggregate(self, u_pred, u_true):
        n_samples = u_pred.shape[0]
        for key in self.keys:
            idx = KEY_TO_INDEX[self.dataset_name][key]
            self.dic[f"{key}_{self.mode}_mse"] += (
                (u_pred[..., idx, :] - u_true[..., idx, :])**2).mean()*n_samples

    def get_dic(self):
        dic = self.dic
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] /= self.n_trajectories
        return self.dic  

@hydra.main(config_path="config/", config_name="ode.yaml")
def main(cfg: DictConfig) -> None:
    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    gamma_step = cfg.optim.gamma_step
    epochs = cfg.optim.epochs

    # inr
    load_run_name = cfg.inr.run_name
    try:
        load_run_dict = dict(cfg.inr.run_dict)
    except TypeError:
        load_run_dict = cfg.inr.run_dict

    inner_steps = cfg.inr.inner_steps

    # dynamics
    model_type = cfg.dynamics.model_type
    hidden = cfg.dynamics.width
    depth = cfg.dynamics.depth
    epsilon = cfg.dynamics.teacher_forcing_init
    epsilon_t = cfg.dynamics.teacher_forcing_decay
    epsilon_freq = cfg.dynamics.teacher_forcing_update

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )

    print("run dir given", run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )

    if run_dir is not None:
        os.symlink(run.dir.split("/files")[0], run_dir)

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name

    print("id", run.id)
    print("dir", run.dir)

    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name

    if data_to_encode is not None:
        model_dir = (
            Path(os.getenv("WANDB_DIR")) /
            dataset_name / data_to_encode / "model"
        )
    else:
        model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "model"

    os.makedirs(str(model_dir), exist_ok=True)

    latent_dim=128
    multichannel = False
    # we need the latent dim and the sub_tr used for training
    # if load_run_name is not None:
    #     multichannel = False
    #     tmp = torch.load(root_dir / "inr" / f"{load_run_name}.pt")
    #     latent_dim = tmp["cfg"].inr.latent_dim
    #     try:
    #         sub_from = tmp["cfg"].data.sub_from
    #     except:
    #         sub_from = 1

    #     sub_tr = tmp["cfg"].data.sub_tr
    #     seed = tmp["cfg"].data.seed

    # elif load_run_dict is not None:
    #     multichannel = True
    #     tmp_data_to_encode = list(load_run_dict.keys())[0]
    #     tmp_run_name = list(load_run_dict.values())[0]
    #     tmp = torch.load(root_dir / tmp_data_to_encode /
    #                      "inr" / f"{tmp_run_name}.pt")
    #     latent_dim = tmp["cfg"].inr.latent_dim
    #     sub_tr = tmp["cfg"].data.sub_tr
    #     seed = tmp["cfg"].data.seed
    #     try:
    #         sub_from = tmp["cfg"].data.sub_from
    #     except:
    #         sub_from = 1

    set_seed(seed)

    (u_train, u_train_eval, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
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
    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_train_eval: {u_train_eval.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}")

    if data_to_encode == None:
        run.tags = (
            ("ode-regression",) + (model_type,) +
            (dataset_name,) + (f"sub={sub_tr}",)
        )
    else:
        run.tags = (
            ("ode-regression",)
            + (model_type,)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
        )

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, latent_dim, dataset_name, data_to_encode
    )
    trainset_extra = TemporalDatasetWithCode(
        u_train_eval, grid_tr_extra, latent_dim, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, latent_dim, dataset_name, data_to_encode
    )

    mean, sigma = u_train.mean(), u_train.std()
    u_train = (u_train - mean) / sigma
    u_test = (u_test - mean) / sigma

    #total frames trainset
    ntrain = trainset.z.shape[0]

    #total frames testset
    ntest = testset.z.shape[0]

    # sequence length 
    T_train = u_train.shape[-1]
    T_test = u_test.shape[-1]

    dt = 1
    timestamps_train = torch.arange(0, T_train, dt).float().cuda()
    timestamps_test = torch.arange(0, T_test, dt).float().cuda()

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = grid_tr.shape[-2]
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = u_train.shape[-2]

    # if load_run_name is not None:
    #     inr, alpha = load_inr_model(
    #         root_dir / "inr",
    #         load_run_name,
    #         data_to_encode,
    #         input_dim=input_dim,
    #         output_dim=output_dim,
    #     )
    #     modulations = load_dynamics_modulations(
    #         trainset,
    #         trainset_extra,
    #         testset,
    #         inr,
    #         root_dir / "modulations",
    #         load_run_name,
    #         inner_steps=inner_steps,
    #         alpha=alpha,
    #         batch_size=2,
    #         data_to_encode=None,
    #         try_reload=False,
    #     )
    #     z_train = modulations["z_train"]
    #     z_train_extra = modulations["z_train_extra"]
    #     z_test = modulations["z_test"]
    #     z_mean = einops.rearrange(z_train, "b l t -> (b t) l").mean(0).reshape(1, latent_dim, 1)
    #     z_std = einops.rearrange(z_train, "b l t -> (b t) l").std(0).reshape(1, latent_dim, 1)
    #     z_train = (z_train - z_mean) / z_std
    #     z_train_extra = (z_train_extra - z_mean) / z_std
    #     z_test = (z_test - z_mean) / z_std

    # elif load_run_dict is not None:
    #     inr_dict = {}
    #     z_mean = {}
    #     z_std = {}
    #     c = len(list(load_run_dict.keys()))
    #     z_train = torch.zeros(ntrain, latent_dim, c, T_train)
    #     z_train_extra = torch.zeros(ntrain, latent_dim, c, T_test)
    #     z_test = torch.zeros(ntest, latent_dim, c, T_test)

    #     for to_encode in list(load_run_dict.keys()):
    #         tmp_name = load_run_dict[to_encode]
    #         #print('tmp name', to_encode, tmp_name)
    #         output_dim = 1
    #         inr, alpha = load_inr_model(
    #             root_dir / to_encode / "inr",
    #             tmp_name,
    #             to_encode,
    #             input_dim=input_dim,
    #             output_dim=output_dim,
    #         )

    #         trainset.set_data_to_encode(to_encode)
    #         trainset_extra.set_data_to_encode(to_encode)
    #         testset.set_data_to_encode(to_encode)

    #         modulations = load_dynamics_modulations(
    #             trainset,
    #             trainset_extra,
    #             testset,
    #             inr,
    #             root_dir / to_encode / "modulations",
    #             tmp_name,
    #             inner_steps=inner_steps,
    #             alpha=alpha,
    #             batch_size=2,
    #             data_to_encode=to_encode,
    #             try_reload=False,
    #         )
    #         inr_dict[to_encode] = inr
    #         z_tr = modulations["z_train"]
    #         z_tr_extra = modulations["z_train_extra"]
    #         z_te = modulations["z_test"]
    #         z_m = einops.rearrange(z_tr, "b l t -> (b t) l").mean(0).reshape(1, latent_dim, 1) #.unsqueeze(0).unsqueeze(-1).repeat(1, 1, T)
    #         z_s = einops.rearrange(z_tr, "b l t -> (b t) l").std(0).reshape(1, latent_dim, 1) #.unsqueeze(0).unsqueeze(-1)
    #         z_mean[to_encode] = z_m
    #         z_std[to_encode] = z_s
    #         z_train[..., KEY_TO_INDEX[dataset_name]
    #                 [to_encode], :] = (z_tr - z_m) / z_s
    #         z_train_extra[..., KEY_TO_INDEX[dataset_name]
    #                 [to_encode], :] = (z_tr_extra - z_m) / z_s
    #         z_test[..., KEY_TO_INDEX[dataset_name]
    #                [to_encode], :] = (z_te - z_m) / z_s

    #     # concat the code
    #     trainset.set_data_to_encode(None)
    #     trainset_extra.set_data_to_encode(None)
    #     testset.set_data_to_encode(None)
    #     # rename inr_dict <- inr
    #     inr = inr_dict

    # trainset.z = z_train
    # trainset_extra.z = z_train_extra
    # testset.z = z_test

    # print('ztrain', z_train.shape, z_train.mean(), z_train.std())
    # print('ztrain_extra', z_train_extra.shape, z_train_extra.mean(), z_train_extra.std())
    # print('ztest', z_test.shape, z_test.mean(), z_test.std())

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    train_extra_loader = torch.utils.data.DataLoader(
        trainset_extra,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
    )
    num_channels = trainset[0][0].shape[-2]
    modes=12
    width=32
    # c = z_train.shape[2] if multichannel else 1 
    model = Derivative(modes, modes, width, num_channels).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        print(f'load {cfg.wandb.checkpoint_path}')
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        # alpha = checkpoint['alpha']
        # best_loss = checkpoint['loss']
        # cfg = checkpoint['cfg']
        # print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        # best_loss = np.inf
    print("epoch_start", epoch_start)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma_step,
        patience=250,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )

    best_loss = np.inf

    if multichannel:
        detailed_train_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                         dataset_name,
                                         mode="train",
                                         n_trajectories=ntrain)
        detailed_train_eval_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                         dataset_name,
                                         mode="train_extra",
                                         n_trajectories=ntrain)
        detailed_test_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                        dataset_name,
                                        mode="test",
                                        n_trajectories=ntest)
    else:
        detailed_train_mse = None
        detailed_train_eval_mse = None
        detailed_test_mse = None
    
    evaluate = cfg.wandb.evaluate
    if evaluate:
        code_train_inter_mse, code_train_extra_mse, total_pred_train_mse = batch_eval_loop(
                model, train_extra_loader,
                timestamps_test, detailed_train_eval_mse,
                ntrain, multichannel,
                dataset_name, T_train, visual_first=4, visual_path=os.path.join(run.dir, f'train_{epoch_start}')
            )

        code_test_inter_mse, code_test_extra_mse, total_pred_test_mse = batch_eval_loop(
                    model, test_loader,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel,
                    dataset_name, T_train, visual_first=4, visual_path=os.path.join(run.dir, f'test_{epoch_start}')
                )
        print(f'train inter mse {code_train_inter_mse}, train_extra_mse {code_train_extra_mse}, test inter mse {code_test_inter_mse}, test extra mse {code_test_extra_mse}')
        return 
    
    for step in range(epochs):
        step_show = step % 100 == 0
        step_show_last = step == epochs - 1

        if step % epsilon_freq == 0:
            epsilon_t = epsilon_t * epsilon

        pred_train_mse = 0
        code_train_mse = 0

        for substep, (images, modulations, coords, idx) in enumerate(train_loader):
            model.train()
            images = images.cuda()
            # modulations = modulations.cuda()
            modulations = images 
            coords = coords.cuda()
            n_samples = images.shape[0]

            # if multichannel:
            #     modulations = einops.rearrange(modulations, "b l c t -> b (l c) t")

            z_pred = ode_scheduling(odeint, model, modulations, timestamps_train, epsilon_t)
            loss = ((z_pred - modulations) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            # if True in (step_show, step_show_last):

            #     pred = get_reconstructions(
            #         inr, coords, z_pred, z_mean, z_std, dataset_name
            #     )
            #     pred_train_mse += ((pred - images) ** 2).mean() * n_samples
                
            #     if multichannel:
            #         detailed_train_mse.aggregate(pred, images)

        code_train_mse = code_train_mse / ntrain

        # if True in (step_show, step_show_last):
        #     pred_train_mse = pred_train_mse / ntrain

        scheduler.step(code_train_mse)

        if T_train != T_test:
            code_train_inter_mse, code_train_extra_mse, total_pred_train_mse = batch_eval_loop(
                model, train_extra_loader,
                timestamps_test, detailed_train_eval_mse,
                ntrain, multichannel,
                dataset_name, T_train
            )

        if True in (step_show, step_show_last):
            if T_train != T_test:
                code_test_inter_mse, code_test_extra_mse, total_pred_test_mse = batch_eval_loop(
                    model, test_loader,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel,
                    dataset_name, T_train
                )

                log_dic = {
                    "pred_train_inter_mse": code_train_inter_mse,
                    "pred_train_extra_mse": code_train_extra_mse,
                    "pred_test_mse_inter": code_test_inter_mse,
                    "pred_test_mse_extra": code_test_extra_mse,
                    # "pred_test_mse": pred_test_mse,
                    # "code_train_inter_mse": code_train_inter_mse,
                    # "code_train_extra_mse": code_train_extra_mse,
                    # "code_test_inter_mse": code_test_inter_mse,
                    # "code_test_extra_mse": code_test_extra_mse,
                }
                
                # if multichannel:

                #     dic_train_mse = detailed_train_mse.get_dic()
                #     detailed_train_mse.reset_dic()

                #     dic_train_extra_mse = detailed_train_eval_mse.get_dic()
                #     detailed_train_eval_mse.reset_dic()

                #     dic_test_mse = detailed_test_mse.get_dic()
                #     detailed_test_mse.reset_dic()

                #     log_dic.update(dic_train_mse)
                #     log_dic.update(dic_train_extra_mse)
                #     log_dic.update(dic_test_mse)

            elif T_train == T_test:
                raise NotImplementedError
                pred_test_mse, code_test_mse, detailed_test_mse = batch_eval_loop(
                    model, inr, test_loader,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel, z_mean, z_std,
                    dataset_name, None
                )
                log_dic = {
                    "pred_train_mse": pred_train_mse,
                    "pred_test_mse": pred_test_mse,
                    "code_train_mse": code_train_mse,
                    "code_test_mse": code_test_mse,
                }
                if multichannel:
                    dic_train_mse = detailed_train_mse.get_dic()
                    detailed_train_mse.reset_dic()

                    dic_test_mse = detailed_test_mse.get_dic()
                    detailed_test_mse.reset_dic()

                    log_dic.update(dic_train_mse)
                    log_dic.update(dic_test_mse)
            wandb.log(log_dic)

        else:
            wandb.log(
                {
                    # "pred_train_inter_mse": pred_train_inter_mse,
                    # "pred_train_extra_mse": pred_train_extra_mse,
                    # "code_train_inter_mse": code_train_inter_mse,
                    # "code_train_extra_mse": code_train_extra_mse,
                    # "pred_train_mse": pred_train_mse,
                    "code_train_mse": code_train_mse,
                },
                step=step,
                commit=not step_show,
            )

        if total_pred_train_mse < best_loss:
            best_loss = total_pred_train_mse
            if T_train != T_test:
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss_inter": code_test_inter_mse,
                        "loss_extra": code_test_extra_mse,
                        # "alpha": alpha,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                    },
                    f"{model_dir}/{run_name}.pt",
                )
            if T_train == T_test:
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # "loss": code_test_mse,
                        # "alpha": alpha,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                    },
                    f"{model_dir}/{run_name}.pt",
                )
        if T_train != T_test:
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_inter": code_test_inter_mse,
                    "loss_extra": code_test_extra_mse,
                    # "alpha": alpha,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{model_dir}/{run_name}_ck.pt",
            )
        if T_train == T_test:
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "loss": code_test_mse,
                    # "alpha": alpha,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{model_dir}/{run_name}_ck.pt",
            )
    return total_pred_train_mse
    
if __name__ == "__main__":
    main()