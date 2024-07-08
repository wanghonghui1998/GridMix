import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo
import math 

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf

from coral.losses import batch_mse_rel_fn
from coral.metalearning import outer_step_metagrid_same_coords_sep_lr, outer_step_metagrid_same_coords_sep_lr_two_stage, outer_step_boosting, outer_step_metagrid_same_coords_teacher_boosting, outer_step_metagrid, outer_step, outer_step_metagrid_same_coords, outer_step_metagrid_part_coords, outer_step_metagrid_twoview, outer_step_metagrid_twoview_rand, outer_step_test_on_diff_grid
from coral.mlp import MLP, Derivative, ResNet
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, rearrange
from coral.utils.data.load_data import get_dynamics_data, set_seed, get_dynamics_data_two_grid, shape2coordinates
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.plot import show
from coral.utils.init_environ import init_environ 

from visualize import write_image_pair, write_image  

@hydra.main(config_path="config/", config_name="siren_metagrid.yaml")
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
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    lr_grid = cfg.optim.lr_grid

    update_modulations=cfg.optim.update_modulations
    random_init=cfg.optim.random_init
    grid_ratio = cfg.optim.grid_ratio
    two_view = cfg.optim.two_view
    two_view_rand = cfg.optim.two_view_rand
    meta_same_grid = cfg.optim.meta_same_grid
    meta_part_grid = cfg.optim.meta_part_grid
    meta_same_grid_sep_lr = cfg.optim.meta_same_grid_sep_lr
    meta_same_grid_sep_lr_two_stage = cfg.optim.meta_same_grid_sep_lr_two_stage
    teacher_boosting = cfg.optim.teacher_boosting
    teacher_ema = cfg.optim.teacher_ema
    extra_only = cfg.optim.extra_only
    w_mod_con = cfg.optim.w_mod_con
    
    sep_lr = meta_same_grid_sep_lr or meta_same_grid_sep_lr_two_stage
    if two_view:
        assert grid_ratio == 0.5
    # print(epochs)
    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim
    code_dim = 0
    if "grid" in model_type:
        grid_channel = cfg.inr.depth - 1
        if cfg.inr.modulate_scale and cfg.inr.modulate_shift:
            grid_channel *= 2
        grid_size = cfg.inr.grid_size
        code_dim = latent_dim - grid_channel * grid_size * grid_size
        
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
    
    regular_grid = shape2coordinates([64,64])
    regular_grid = regular_grid.reshape(1, 64*64, 2).cuda()
    print(regular_grid.shape)
    set_seed(seed)
    (u_train, u_train_out1, u_train_out2, u_eval_extrapolation, u_eval_extrapolation_in, u_test, u_test_in, grid_tr, grid_tr_out1, grid_tr_out2, grid_tr_extra, grid_tr_extra_in, grid_te, grid_te_in) = get_dynamics_data_two_grid(
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

    # (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
    #     data_dir,
    #     dataset_name,
    #     ntrain,
    #     ntest,
    #     seq_inter_len=seq_inter_len,
    #     seq_extra_len=seq_extra_len,
    #     sub_from=sub_from,
    #     sub_tr=sub_tr,
    #     sub_te=sub_te,
    #     same_grid=same_grid,
    # )
    print(f"data: {dataset_name}, u_train: {u_train.shape}, u_eval_extrapolation: {u_eval_extrapolation.shape}, u_test: {u_test.shape}, u_test_in: {u_test_in.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}, grid_te_in: {grid_te_in.shape}")
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
    testset_in = TemporalDatasetWithCode(
        u_test_in, grid_te_in, latent_dim, dataset_name, data_to_encode
    )
    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = trainset.input_dim
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = trainset.output_dim

    # transforms datasets shape into (N * T, Dx, Dy, C)
    trainset = rearrange(trainset, dataset_name)
    testset = rearrange(testset, dataset_name)
    testset_in = rearrange(testset_in, dataset_name)
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
    test_in_loader = torch.utils.data.DataLoader(
        testset_in,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
    ) 

    set_seed(seed)
    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )

    if cfg.distributed:
        inr = torch.nn.parallel.DistributedDataParallel(inr, device_ids=[cfg.gpu_id])
        inr_without_ddp = inr.module 
    else:
        inr_without_ddp = inr

    if teacher_boosting:
        teacher_inr = create_inr_instance(
            cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
        )
        teacher_inr.requires_grad = False
        for param in teacher_inr.parameters():
            param.requires_grad = False
        teacher_inr.load_state_dict(inr.state_dict())
    else:
        teacher_inr = None

    print(inr)
    if random_init:
        random_init_std = 0.001 * math.sqrt(1.0 / cfg.inr.hidden_dim)
    alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))
    if sep_lr:
        alpha_grid = nn.Parameter(torch.Tensor([lr_code]).to(device))
    else:
        alpha_grid = torch.Tensor([0.0])
    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code
    
    update_alpha = cfg.optim.update_alpha
    if update_alpha:
        if sep_lr:
            optimizer = torch.optim.AdamW(
                [
                    {"params": inr.parameters(), "lr": lr_inr},
                    {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
                    {"params": alpha_grid, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
                ],
                lr=lr_inr,
                weight_decay=0,
            )
        else:
            if 'MoG' in model_type:
                params_net = []
                params_grid = []
                for name, param in inr.named_parameters():
                    if 'grid_base' in name:
                        params_grid.append(param)
                    else:
                        params_net.append(param)

                optimizer = torch.optim.AdamW(
                [
                    {"params": params_net, "lr": lr_inr},
                    {"params": params_grid, "lr": lr_grid},
                    {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
                ],
                lr=lr_inr,
                weight_decay=0,
            )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": inr.parameters(), "lr": lr_inr},
                        {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
                    ],
                    lr=lr_inr,
                    weight_decay=0,
                )
    else:
        # print('not optmize alpha')
        optimizer = torch.optim.AdamW(
            [
                {"params": inr.parameters(), "lr": lr_inr},
                # {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
            ],
            lr=lr_inr,
            weight_decay=0,
        )

    if saved_checkpoint:
        inr.load_state_dict(checkpoint['inr'])
        optimizer.load_state_dict(checkpoint['optimizer_inr'])
        epoch_start = checkpoint['epoch']
        alpha = checkpoint['alpha']
        if sep_lr:
            alpha_grid = checkpoint['alpha_grid']
        best_loss = checkpoint['loss']
        # cfg = checkpoint['cfg']
        epochs = epoch_start + 1
        print("epoch_start, alpha, best_loss", epoch_start, alpha.item(), best_loss)
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

    save_per_plot = False 
    plot = 'grid' in model_type
    visual_mod = cfg.inr.grid_size
    plot_frame = 40 

    rel_train_mse = torch.tensor([0.0]).to(device)
    fit_train_mse = torch.tensor([0.0]).to(device)
    loss_mod_con = torch.tensor([0.0]).to(device)
    for step in range(epoch_start, epochs):
        rel_train_mse[0] = 0
        rel_test_mse = 0
        rel_test_mse_train_coords = 0
        fit_train_mse[0] = 0
        fit_test_mse = 0
        fit_test_mse_train_coords = 0
        diff_fit_test_mse_train_coords = 0
        use_rel_loss = step % 10 == 0
        step_show = step % 100 == 0
        step_show_last = step == epochs - 1
        loss_mod_con[0] = 0
        test_loss_mod_con = 0

        if cfg.distributed:
            train_loader.sampler.set_epoch(step)

        for substep, (images, modulations, coords, idx) in enumerate(train_loader):
            # print(alpha)
            inr.train()
            images = images.to(device)  # torch.Size([128, 64, 64, 1])
            modulations = modulations.to(device)    # torch.Size([128, 128])
            coords = coords.to(device)  # torch.Size([128, 64, 64, 2])         
            # print(images.shape, modulations.shape, coords.shape)    
            if coords.dim() == 4:
                coords = coords.reshape(coords.shape[0], -1, coords.shape[-1])
                images = images.reshape(images.shape[0], -1, images.shape[-1])

            n_samples = images.shape[0]
            # print(images.shape)
            if not update_modulations:
                if random_init:
                    input_modulations = random_init_std * torch.randn_like(modulations)
                else:
                    input_modulations = torch.zeros_like(modulations)
            else:
                input_modulations = modulations
            # import pdb; pdb.set_trace()
            # print(input_modulations.mean())
            if two_view:
                outputs = outer_step_metagrid_twoview(
                    inr,
                    coords,
                    images,
                    inner_steps,
                    alpha,
                    grid_ratio,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    # modulations=torch.zeros_like(modulations),
                    modulations=input_modulations,
                )
                # loss_mod_con += outputs["loss_mod_con"].cpu().detach()
            elif two_view_rand:
                outputs = outer_step_metagrid_twoview_rand(
                    inr,
                    coords,
                    images,
                    inner_steps,
                    alpha,
                    grid_ratio,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    # modulations=torch.zeros_like(modulations),
                    modulations=input_modulations,
                    code_dim=code_dim,
                    grid_channel=grid_channel,
                    grid_size=grid_size,
                )
            
            elif meta_same_grid:
                if teacher_boosting:
                    outputs = outer_step_metagrid_same_coords_teacher_boosting(
                        inr,
                        coords,
                        images,
                        teacher_inr,
                        regular_grid.repeat(n_samples, 1, 1),
                        inner_steps,
                        alpha,
                        grid_ratio,
                        is_train=True,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        extra_only=extra_only,
                    )
                else:
                    outputs = outer_step_metagrid_same_coords(
                        inr,
                        coords,
                        images,
                        inner_steps,
                        alpha,
                        grid_ratio,
                        is_train=True,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                    )
            elif meta_same_grid_sep_lr:
                outputs = outer_step_metagrid_same_coords_sep_lr(
                        inr,
                        coords,
                        images,
                        inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio,
                        is_train=True,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
            elif meta_same_grid_sep_lr_two_stage:
                outputs = outer_step_metagrid_same_coords_sep_lr_two_stage(
                        inr,
                        coords,
                        images,
                        inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio,
                        is_train=True,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
            elif meta_part_grid:
                outputs = outer_step_metagrid_part_coords(
                    inr,
                    coords,
                    images,
                    inner_steps,
                    alpha,
                    grid_ratio,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    # modulations=torch.zeros_like(modulations),
                    modulations=input_modulations,
                )
                
            else:
                outputs = outer_step_metagrid(
                    inr,
                    coords,
                    images,
                    inner_steps,
                    alpha,
                    grid_ratio,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                    # modulations=torch.zeros_like(modulations),
                    modulations=input_modulations,
                )

            optimizer.zero_grad()
            if w_mod_con > 0:
                total_loss = outputs["loss"] + w_mod_con * outputs["loss_mod_con"]
                loss_mod_con += outputs["loss_mod_con"].detach().item() * n_samples
            else:
                total_loss = outputs["loss"] 
            total_loss.backward(create_graph=False)
            # outputs["loss"].backward(create_graph=False)
            if 'fno' not in model_type:
                nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            optimizer.step()
            if teacher_boosting:
                if teacher_ema > 0:
                    for param_q, param_k in zip(inr.parameters(), teacher_inr.parameters()):
                        param_k.data = param_k.data * teacher_ema + param_q.data * (1. - teacher_ema)
                else:
                    teacher_inr.load_state_dict(inr.state_dict())
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

            if update_modulations:
                trainset[idx] = outputs["modulations"].detach().cpu()
                # modulations = outputs["modulations"].detach()
            # subsubstep += 1
            # print(f'{subsubstep} alpha {alpha.item()}, loss {loss.item()}')
            # import imageio.v2 as imageio
            # modulations = modulations.reshape(1,6,64,64,1).cpu().numpy()
            # for image_idx in range(6):
            #     gamma = modulations[0,image_idx]
            #     imageio.imwrite(os.path.join(run.dir, f'{image_idx}.png'), (255*(gamma-gamma.min())/(gamma.max()-gamma.min())).astype(np.uint8))
            # return
        if cfg.distributed:
            dist.all_reduce(fit_train_mse, op=dist.ReduceOp.SUM, async_op=False)
        train_loss = fit_train_mse / ntrain
        if w_mod_con > 0:
            if cfg.distributed:
                dist.all_reduce(loss_mod_con, op=dist.ReduceOp.SUM, async_op=False)
            loss_mod_con = loss_mod_con / ntrain
        if model_type=="fourier_features":
            scheduler.step(train_loss)

        if use_rel_loss:
            if cfg.distributed:
                dist.all_reduce(rel_train_mse, op=dist.ReduceOp.SUM, async_op=False)
            rel_train_loss = rel_train_mse / ntrain

        if cfg.rank == 0 and True in (step_show, step_show_last):
            plot_modulations = []
            plot_modulations_num = 0
            all_modulations = []
            for images, modulations, coords, idx in test_loader:
                inr.eval()
                images = images.to(device)
                modulations = modulations.to(device)
                coords = coords.to(device)
                n_samples = images.shape[0]

                if random_init:
                    input_modulations = random_init_std * torch.randn_like(modulations)
                else:
                    input_modulations = torch.zeros_like(modulations)
                if teacher_boosting:
                    outputs = outer_step_metagrid_same_coords_teacher_boosting(
                        inr,
                        coords,
                        images,
                        teacher_inr,
                        regular_grid.repeat(n_samples, 1, 1),
                        test_inner_steps,
                        alpha,
                        1,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        extra_only=extra_only,
                    )
                elif meta_same_grid_sep_lr:
                    outputs = outer_step_metagrid_same_coords_sep_lr(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio=1.0,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
                elif meta_same_grid_sep_lr_two_stage:
                    outputs = outer_step_metagrid_same_coords_sep_lr_two_stage(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio=1.0,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
                else:
                    outputs = outer_step(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                    )
                # outputs = outer_step_boosting(
                #     inr,
                #     regular_grid.repeat(n_samples, 1, 1),
                #     coords,
                #     images,
                #     test_inner_steps,
                #     alpha,
                #     is_train=False,
                #     return_reconstructions=False,
                #     gradient_checkpointing=False,
                #     use_rel_loss=use_rel_loss,
                #     loss_type="mse",
                #     # modulations=torch.zeros_like(modulations),
                #     modulations=input_modulations,
                # )
                # diff_images = test_in_loader.dataset.v[idx]
                # diff_coords = test_in_loader.dataset.c[idx]
                # diff_images = diff_images.to(device)
                # diff_coords = diff_coords.to(device)
                # outputs = outer_step_test_on_diff_grid(
                #     inr,
                #     coords,
                #     images,
                #     diff_coords, 
                #     diff_images,
                #     test_inner_steps,
                #     alpha,
                #     is_train=False,
                #     return_reconstructions=False,
                #     gradient_checkpointing=False,
                #     use_rel_loss=use_rel_loss,
                #     loss_type="mse",
                #     # modulations=torch.zeros_like(modulations),
                #     modulations=input_modulations,
                # )
                # diff_fit_test_mse_train_coords += outputs["diff_loss"].item() * n_samples
                if plot and plot_modulations_num < plot_frame:
                    plot_modulations.append(outputs["modulations"][:plot_frame-plot_modulations_num])
                    plot_modulations_num += plot_modulations[-1].shape[0]
                loss = outputs["loss"]
                fit_test_mse += loss.item() * n_samples
                # all_modulations.append(outputs["modulations"][...,code_dim:].detach())
                all_modulations.append(outputs["modulations"].detach())
                if use_rel_loss:
                    rel_test_mse += outputs["rel_loss"].item() * n_samples

            test_loss = fit_test_mse / ntest
            diff_test_loss_train_coords= diff_fit_test_mse_train_coords / ntest
            if use_rel_loss:
                rel_test_loss = rel_test_mse / ntest

            plot_modulations_train_coords = []
            plot_modulations_num = 0
            all_modulations_in = []
            for images, modulations, coords, idx in test_in_loader:
                inr.eval()
                images = images.to(device)
                modulations = modulations.to(device)
                coords = coords.to(device)
               
                n_samples = images.shape[0]

                if random_init:
                    input_modulations = random_init_std * torch.randn_like(modulations)
                else:
                    input_modulations = torch.zeros_like(modulations)
                if teacher_boosting:
                    outputs = outer_step_metagrid_same_coords_teacher_boosting(
                        inr,
                        coords,
                        images,
                        teacher_inr,
                        regular_grid.repeat(n_samples, 1, 1),
                        test_inner_steps,
                        alpha,
                        1,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        extra_only=extra_only,
                    )
                elif meta_same_grid_sep_lr:
                    outputs = outer_step_metagrid_same_coords_sep_lr(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio=1.0,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
                elif meta_same_grid_sep_lr_two_stage:
                    outputs = outer_step_metagrid_same_coords_sep_lr_two_stage(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        alpha_grid,
                        grid_ratio=1.0,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                        code_dim=code_dim,
                    )
                else:
                    outputs = outer_step(
                        inr,
                        coords,
                        images,
                        test_inner_steps,
                        alpha,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                        # modulations=torch.zeros_like(modulations),
                        modulations=input_modulations,
                    )
                # outputs = outer_step_boosting(
                #     inr,
                #     regular_grid.repeat(n_samples, 1, 1),
                #     coords,
                #     images,
                #     test_inner_steps,
                #     alpha,
                #     is_train=False,
                #     return_reconstructions=False,
                #     gradient_checkpointing=False,
                #     use_rel_loss=use_rel_loss,
                #     loss_type="mse",
                #     # modulations=torch.zeros_like(modulations),
                #     modulations=input_modulations,
                # )
                
                if plot and plot_modulations_num < plot_frame:
                    plot_modulations_train_coords.append(outputs["modulations"][:plot_frame-plot_modulations_num])
                    plot_modulations_num += plot_modulations_train_coords[-1].shape[0]
                loss = outputs["loss"]
                fit_test_mse_train_coords += loss.item() * n_samples
                
                # all_modulations_in.append(outputs["modulations"][...,code_dim:].detach())
                all_modulations_in.append(outputs["modulations"].detach())
                if use_rel_loss:
                    rel_test_mse_train_coords += outputs["rel_loss"].item() * n_samples
            
            test_loss_train_coords = fit_test_mse_train_coords / ntest
            
            if use_rel_loss:
                rel_test_loss_train_coords = rel_test_mse_train_coords / ntest
            if "grid" in model_type:
                all_modulations_in = torch.cat(all_modulations_in, dim=0)
                all_modulations_in_code = all_modulations_in[...,:code_dim]
                all_modulations_in_grid = all_modulations_in[...,code_dim:].reshape(-1, grid_channel, grid_size*grid_size)
                all_modulations = torch.cat(all_modulations, dim=0)
                all_modulations_code = all_modulations[...,:code_dim]
                all_modulations_grid = all_modulations[...,code_dim:].reshape(-1, grid_channel, grid_size*grid_size)
                test_loss_code_con = torch.mean(torch.linalg.norm(all_modulations_code-all_modulations_in_code, 2, dim=-1) / (torch.linalg.norm(all_modulations_in_code, 2, dim=-1) + 1e-10))
                test_loss_grid_con = torch.mean(torch.linalg.norm(all_modulations_grid-all_modulations_in_grid, 2, dim=-1) / (torch.linalg.norm(all_modulations_in_grid, 2, dim=-1) + 1e-10))

            else:
                all_modulations_in = torch.cat(all_modulations_in, dim=0)
                all_modulations = torch.cat(all_modulations, dim=0)
                test_loss_code_con = torch.mean(torch.linalg.norm(all_modulations-all_modulations_in, 2, dim=-1) / (torch.linalg.norm(all_modulations_in, 2, dim=-1) + 1e-10))
                test_loss_grid_con = 0.0
            if plot:
                plot_modulations = torch.cat(plot_modulations, dim=0)
                plot_modulations_train_coords = torch.cat(plot_modulations_train_coords, dim=0)
                modulations_v = plot_modulations[...,code_dim:].reshape(plot_modulations.shape[0], -1, visual_mod, visual_mod)
                modulations_v_train_coords = plot_modulations_train_coords[...,code_dim:].reshape(plot_modulations_train_coords.shape[0], -1, visual_mod, visual_mod)
                # divider = 2 * modulations_v.shape[1]
                # modulations_v = modulations_v.permute(1,0,2,3).reshape(-1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                # visual_path = os.path.join(run.dir, f'testmod_{step}.png') if save_per_plot else os.path.join(run.dir, f'testmod.png') 
                # write_image(modulations_v, modulations_v, 0, path=visual_path, cmap='twilight_shifted', divider=divider)

                divider = 2 
                modulations_v = modulations_v.permute(1,0,2,3).unsqueeze(-1).detach().cpu().numpy()
                modulations_v_train_coords = modulations_v_train_coords.permute(1,0,2,3).unsqueeze(-1).detach().cpu().numpy()
                for layer_idx in range(modulations_v.shape[0]):
                    visual_path = os.path.join(run.dir, f'testmod_l{layer_idx}_{step}.png') if save_per_plot else os.path.join(run.dir, f'testmod_l{layer_idx}.png') 
                    # write_image(modulations_v[layer_idx], modulations_v[layer_idx], 0, path=visual_path, cmap='twilight_shifted', divider=divider)
                    write_image_pair(modulations_v[layer_idx], modulations_v_train_coords[layer_idx], 0, path=visual_path, cmap='twilight_shifted', divider=divider)

        if cfg.rank == 0:
            if True in (step_show, step_show_last):
                print(f'{step} alpha {alpha.item():.4e}, alpha_grid {alpha_grid.item():.4e}, train loss {train_loss.item():.4e}, test loss {test_loss:.4e},  test loss on diff test coords {diff_test_loss_train_coords:.4e}, test loss on train coords {test_loss_train_coords:.4e}, train loss_mod_con: {loss_mod_con.item():.4e}, test loss_code_con: {test_loss_code_con:.4e}, test loss_grid_con: {test_loss_grid_con:.4e}')
                wandb.log(
                    {
                        "test_rel_loss": rel_test_loss,
                        "train_rel_loss": rel_train_loss,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                    },
                )
                # print(f'ep {step}: train loss: {train_loss}, test loss: {test_loss}')

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

                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr": inr.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": best_loss,
                        "alpha": alpha,
                        "alpha_grid": alpha_grid,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )
            if True in (step_show, step_show_last):
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr": inr.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": train_loss,
                        "alpha": alpha,
                        "alpha_grid": alpha_grid,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                    },
                    f"{RESULTS_DIR}/{run_name}_ck.pt",
                )
    return rel_test_loss

if __name__ == "__main__":
    main()