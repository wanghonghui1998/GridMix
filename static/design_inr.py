import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo

sys.path.append(str(Path(__file__).parents[1]))

import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint

from coral.losses import batch_mse_rel_fn
from coral.mfn import FourierNet, HyperMAGNET, HyperMultiscaleBACON
from coral.mlp import MLP, Derivative, ResNet
from coral.siren import ModulatedSiren
from coral.utils.data.load_data import set_seed
from coral.utils.plot import show
from coral.utils.data.load_data import get_operator_data
from coral.utils.data.operator_dataset import OperatorDataset
from coral.utils.models.load_inr import create_inr_instance
import torch.utils.checkpoint as cp
from coral.metalearning import outer_step
from torch_geometric.loader import DataLoader

@hydra.main(config_path="config/static/", config_name="design.yaml")
def main(cfg: DictConfig) -> None:

    torch.set_default_dtype(torch.float32)

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_tr = cfg.data.sub_tr
    seed = cfg.data.seed

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    gamma_step = cfg.optim.gamma_step
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    lr_mlp = cfg.optim.lr_mlp
    weight_decay_mlp = cfg.optim.weight_decay_mlp
    lr_grid = cfg.optim.lr_grid
    optim_out = cfg.optim.optim_out
    optim_in = cfg.optim.optim_in

    # inr
    latent_dim = cfg.inr_in.latent_dim
    in_model_type = cfg.inr_in.model_type
    out_model_type = cfg.inr_out.model_type
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
    sweep_id = cfg.wandb.sweep_id

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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

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
    sweep_id = cfg.wandb.sweep_id

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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = (
        ("different-inr-regression",)
        + (dataset_name,)
        + (f"sub={sub_tr}",)
    )

    set_seed(seed)

    x_train, y_train, x_test, y_test, grid_tr, grid_te = get_operator_data(
    data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True)

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
    print('grid_tr', grid_tr.shape)
    print('grid_te', grid_te.shape)

    trainset = OperatorDataset(x_train,
        y_train,
        grid_tr,
        latent_dim_a=cfg.inr_in.latent_dim,
        latent_dim_u=cfg.inr_out.latent_dim,
        dataset_name=None,
        data_to_encode=None,
    )

    testset = OperatorDataset(x_test,
        y_test,
        grid_te,
        latent_dim_a=cfg.inr_in.latent_dim,
        latent_dim_u=cfg.inr_out.latent_dim,
        dataset_name=None,
        data_to_encode=None,
    )
    ntrain = len(trainset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    print("train", len(trainset))
    print('test', len(testset))

    input_dim = 2
    output_dim_in = 2
    output_dim_out = 1

    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda"
    )
    cfg.inr = cfg.inr_out
    inr_out = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda"
    )
    print(inr_in)
    print(inr_out)
    alpha_in = nn.Parameter(torch.Tensor([lr_code]).cuda())
    alpha_out = nn.Parameter(torch.Tensor([lr_code]).cuda())

    if 'GridMix' in in_model_type:
        in_params_net = []
        in_params_grid = []
        for name, param in inr_in.named_parameters():
            if 'grid_base' in name:
                in_params_grid.append(param)
            else:
                in_params_net.append(param)

        optimizer_in = torch.optim.AdamW(
            [
                {"params": in_params_net, "lr": lr_inr},
                {"params": in_params_grid, "lr": lr_grid},
                {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
            ],
            lr=lr_inr,
            weight_decay=0,
        )
    else:
        optimizer_in = torch.optim.AdamW(
            [
                {"params": inr_in.parameters()},
                {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
            ],
            lr=lr_inr,
            weight_decay=0,
        )

    if 'GridMix' in out_model_type:
        out_params_net = []
        out_params_grid = []
        for name, param in inr_out.named_parameters():
            if 'grid_base' in name:
                out_params_grid.append(param)
            else:
                out_params_net.append(param)

        optimizer_out = torch.optim.AdamW(
            [
                {"params": out_params_net, "lr": lr_inr},
                {"params": out_params_grid, "lr": lr_grid},
                {"params": alpha_out, "lr": meta_lr_code, "weight_decay": 0},
            ],
            lr=lr_inr,
            weight_decay=0,
        )
    else:
        optimizer_out = torch.optim.AdamW(
            [
                {"params": inr_out.parameters()},
                {"params": alpha_out, "lr": meta_lr_code, "weight_decay": 0},
            ],
            lr=lr_inr,
            weight_decay=0,
        )

    best_loss = np.inf
    
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        # "cfg": cfg,
        #                 "epoch": step,
        #                 "inr_in": inr_in.state_dict(),
        #                 "inr_out": inr_out.state_dict(),
        #                 "optimizer_inr_in": optimizer_in.state_dict(),
        #                 "optimizer_inr_out": optimizer_out.state_dict(),
        #                 "loss": test_loss_out,
        #                 "alpha_in": alpha_in,
        #                 "alpha_out": alpha_out,
        print(f'load {cfg.wandb.checkpoint_path}')
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        inr_in.load_state_dict(checkpoint['inr_in'])
        inr_out.load_state_dict(checkpoint['inr_out'])

        optimizer_in.load_state_dict(checkpoint['optimizer_inr_in'])
        optimizer_out.load_state_dict(checkpoint['optimizer_inr_out'])
        alpha_in = checkpoint['alpha_in']
        alpha_out = checkpoint['alpha_out']
        epoch_start = checkpoint['epoch']
        # alpha = checkpoint['alpha']
        # best_loss = checkpoint['loss']
        # cfg = checkpoint['cfg']
        # print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        # best_loss = np.inf
    print("epoch_start", epoch_start)

    for step in range(epoch_start, epochs):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        rel_train_mse_out = 0
        rel_test_mse_out = 0
        fit_train_mse_out = 0
        fit_test_mse_out = 0
        use_pred_loss = step % 10 == 0
        use_pred_loss = step % 20 == 0
        step_show = step % 200 == 0
        for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(
            train_loader
        ):
            inr_in.train()
            inr_out.train()

            a_s = a_s.cuda()
            u_s = u_s.cuda()
            za_s = za_s.cuda()
            zu_s = zu_s.cuda()
            coords = coords.cuda()
            n_samples = a_s.shape[0]

            if optim_in:
                # input
                outputs = outer_step(
                    inr_in,
                    coords,
                    a_s,
                    inner_steps,
                    alpha_in,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_pred_loss,
                    loss_type="mse",
                    modulations=torch.zeros_like(za_s),
                )

                optimizer_in.zero_grad()
                outputs["loss"].backward(create_graph=False)
                nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
                optimizer_in.step()
                loss = outputs["loss"].cpu().detach()
                fit_train_mse_in += loss.item() * n_samples
                z0 = outputs["modulations"].detach()

                # mu0 = beta1 * mu0 + (1-beta1)*z0.mean(0)
                # sigma0 = beta1 * sigma0 + (1-beta1)*z0.std(0) + 1e-8

                if step_show and substep == 0:
                    u_pred = inr_in(coords, z0)
                    with torch.no_grad():
                        show(a_s, u_pred, coords, "train_input", num_examples=4)

            if optim_out:
                # output
                outputs = outer_step(
                    inr_out,
                    coords,
                    u_s,
                    inner_steps,
                    alpha_out,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_pred_loss,
                    loss_type="mse",
                    modulations=torch.zeros_like(za_s),
                )

                optimizer_out.zero_grad()
                outputs["loss"].backward(create_graph=False)
                nn.utils.clip_grad_value_(inr_out.parameters(), clip_value=1.0)
                optimizer_out.step()
                loss = outputs["loss"].cpu().detach()
                fit_train_mse_out += loss.item() * n_samples

                # mlp regression
                z1 = outputs["modulations"].detach()

                # mu1 = beta1 * mu1 + (1-beta1)*z1.mean(0)
                # sigma1 = beta1 * sigma1 + (1-beta1)*z1.std(0) + 1e-8


        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_out = fit_train_mse_out / (ntrain)

        if use_pred_loss:
            rel_train_loss_out = rel_train_mse_out / (ntrain)

        for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(
            test_loader
        ):
            inr_in.eval()
            inr_out.eval()
            a_s = a_s.cuda()
            u_s = u_s.cuda()
            za_s = za_s.cuda()
            zu_s = zu_s.cuda()
            coords = coords.cuda()
            n_samples = a_s.shape[0]

            if optim_in:
                # input
                outputs = outer_step(
                    inr_in,
                    coords,
                    a_s,
                    inner_steps,
                    alpha_in,
                    is_train=False,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_pred_loss,
                    loss_type="mse",
                    modulations=torch.zeros_like(za_s),
                )

                loss = outputs["loss"].cpu().detach()
                fit_test_mse_in += loss.item() * n_samples
                z0 = outputs["modulations"].detach()

                if step_show and substep == 0:
                    u_pred = inr_in(coords, z0)
                    with torch.no_grad():
                        show(a_s, u_pred, coords, "test_input", num_examples=4)
            if optim_out:
                # output
                outputs = outer_step(
                    inr_out,
                    coords, # a_s
                    u_s,
                    inner_steps,
                    alpha_out,
                    is_train=False,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_pred_loss,
                    loss_type="mse",
                    modulations=torch.zeros_like(za_s),
                )

                loss = outputs["loss"].cpu().detach()
                fit_test_mse_out += loss.item() * n_samples
                z1 = outputs["modulations"].detach()

                if use_pred_loss:
                    rel_test_mse_out += outputs["rel_loss"].item() * n_samples

        test_loss_in = fit_test_mse_in / (ntest)
        test_loss_out = fit_test_mse_out / (ntest)

        if use_pred_loss:
            rel_test_loss_out = rel_test_mse_out / (ntest)

        COMMIT = not use_pred_loss
        wandb.log(
            {
                "train_loss_in": train_loss_in,
                "test_loss_in": test_loss_in,
                "train_loss_out": train_loss_out,
                "test_loss_out": test_loss_out,
            },
            step=step,
            commit=COMMIT
        )

        if use_pred_loss:
            COMMIT = True
            wandb.log(
                {
                    "train_rel_loss_out": rel_train_loss_out,
                    "test_rel_loss_out": rel_test_loss_out,
                })

        if optim_out:
            if train_loss_out < best_loss:
                best_loss = train_loss_out

                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr_in": inr_in.state_dict(),
                        "inr_out": inr_out.state_dict(),
                        "optimizer_inr_in": optimizer_in.state_dict(),
                        "optimizer_inr_out": optimizer_out.state_dict(),
                        "loss": test_loss_out,
                        "alpha_in": alpha_in,
                        "alpha_out": alpha_out,
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )
        else:
            if train_loss_in < best_loss:
                best_loss = train_loss_in

                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr_in": inr_in.state_dict(),
                        "inr_out": inr_out.state_dict(),
                        "optimizer_inr_in": optimizer_in.state_dict(),
                        "optimizer_inr_out": optimizer_out.state_dict(),
                        "loss": test_loss_in,
                        "alpha_in": alpha_in,
                        "alpha_out": alpha_out,
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )


    return test_loss_out


if __name__ == "__main__":
    main()
