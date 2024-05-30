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
import torch.distributed as dist
import torch.nn.functional as F

import wandb
from omegaconf import DictConfig, OmegaConf

from coral.losses import batch_mse_rel_fn
from coral.metalearning import outer_step
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

class INRwCODE(nn.Module):
    def __init__(self, inr, t, code, lr_code):
        super().__init__()
        self.inr = inr 
        self.code = nn.parameter.Parameter(torch.zeros(t, code)) # -> (n,t,code)
        # self.n = n 
        self.t = t 
        self.alpha = nn.parameter.Parameter(torch.Tensor([lr_code]))

    # def get_code(self, idx):
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
    lr_edg = cfg.optim.lr_edg
    meta_lr_code = cfg.optim.meta_lr_code
    lr_adapt = cfg.optim.lr_adapt
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    use_cl = cfg.optim.use_cl
    temperature = cfg.optim.temperature
    w_cl = cfg.optim.w_cl

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
    u_test = u_test[..., 0:u_train.shape[-1]]
    grid_te = grid_te[..., 0:u_train.shape[-1]]
    print(f"use the training horizons for test, u_test:{u_test.shape}")
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
    n_frames_code = u_train.shape[-1]
    inr_w_code = INRwCODE(inr, u_train.shape[-1], latent_dim, lr_code).to(device)

    if cfg.distributed:
        inr_w_code = torch.nn.parallel.DistributedDataParallel(inr_w_code, device_ids=[cfg.gpu_id])
        inr_without_ddp = inr_w_code.module 
    else:
        inr_without_ddp = inr_w_code

    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code
    
    params_inr = []
    params_code = []
    params_alpha = []
    for name, param in inr_w_code.named_parameters():
        if 'code' in name:
            params_code.append(param)
        elif 'alpha' in name:
            params_alpha.append(param)
        else:
            params_inr.append(param)

    update_alpha = cfg.optim.update_alpha

    optimizer = torch.optim.AdamW(
        [
            {"params": params_inr, "lr": lr_inr},
            {"params": params_code, "lr": lr_edg}, # use independent lr ? 
            {"params": params_alpha, "lr": meta_lr_code},
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

    if use_cl:
        cl_criterion = torch.nn.CrossEntropyLoss().to(device)

    rel_train_mse = torch.tensor([0.0]).to(device)
    # rel_test_mse = torch.tensor([0.0]).to(device)
    fit_train_mse = torch.tensor([0.0]).to(device)
    cl_train = torch.tensor([0.0]).to(device)
    # fit_test_mse = torch.tensor([0.0]).to(device)
    for step in range(epoch_start, epochs):
        rel_train_mse[0] = 0
        rel_test_mse = 0
        fit_train_mse[0] = 0
        cl_train[0] = 0 
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
            idx = idx % n_frames_code
            input_modulations = inr_without_ddp.code[idx]
            # print(images.shape)
            # if not update_modulations:
            #     input_modulations = torch.zeros_like(modulations)
            # else:
            #     input_modulations = modulations
            #     raise NotImplementedError
            # import pdb; pdb.set_trace()
            # print(modulations.mean())
            outputs = outer_step(
                inr_w_code,
                coords,
                images,
                inner_steps,
                inr_without_ddp.alpha,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
                # modulations=torch.zeros_like(modulations),
                modulations=input_modulations,
            )
            if use_cl:
                # import pdb; pdb.set_trace()
                output_modulations = F.normalize(outputs['modulations'],dim=1) # (b, c)
                c_norm = F.normalize(inr_without_ddp.code,dim=1)    # (t,c)
                similarity_matrix = torch.sum(output_modulations.unsqueeze(1) * c_norm.unsqueeze(0), dim=-1)    # (b, t)
                logits = similarity_matrix / temperature
                labels = idx.to(device)
                loss_cl = cl_criterion(logits, labels)
                cl_train[0] += loss_cl.item() * n_samples

            optimizer.zero_grad()
            if use_cl:
                total_loss = w_cl * loss_cl + outputs["loss"]
                total_loss.backward(create_graph=False)
            else:
                outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_w_code.parameters(), clip_value=1.0)
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
            dist.all_reduce(cl_train, op=dist.ReduceOp.SUM, async_op=False)
        train_loss = fit_train_mse / ntrain
        cl_train_loss = cl_train / ntrain

        if model_type=="fourier_features":
            scheduler.step(train_loss)

        if use_rel_loss:
            if cfg.distributed:
                dist.all_reduce(rel_train_mse, op=dist.ReduceOp.SUM, async_op=False)
            rel_train_loss = rel_train_mse / ntrain

        if cfg.rank == 0 and (True in (step_show, step_show_last)):
            # set_requires_grad(inr_without_ddp, False)
            # save_best = True 
            # assert len(test_loader) == 1
            for images, modulations, coords, idx in test_loader:
                inr_w_code.eval()
                images = images.to(device)
                modulations = modulations.to(device)
                coords = coords.to(device)
                n_samples = images.shape[0]

                idx = idx % n_frames_code
                input_modulations = inr_without_ddp.code[idx].detach()

                outputs = outer_step(
                    inr_w_code,
                    coords,
                    images,
                    test_inner_steps,
                    inr_without_ddp.alpha,
                    is_train=False,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    # modulations=torch.zeros_like(modulations),
                    modulations=input_modulations,
                )

                loss = outputs["loss"]
                fit_test_mse += loss.item() * n_samples
            
                if use_rel_loss:
                    rel_test_mse += outputs["rel_loss"].item() * n_samples

            test_loss = fit_test_mse / ntest

            if use_rel_loss:
                rel_test_loss = rel_test_mse / ntest
            # set_requires_grad(inr_without_ddp, True)

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
                        "cl_train_loss": cl_train_loss,
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