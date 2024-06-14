from coral.utils.plot import show
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.load_inr import create_inr_instance, load_inr_model
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from coral.utils.data.load_modulations import load_dynamics_modulations
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import (KEY_TO_INDEX, TemporalDatasetWithCode)
from coral.mlp import DerivativeFNO, Derivative
from torchdiffeq import odeint
from omegaconf import DictConfig, OmegaConf
import wandb
import torch.nn as nn
import torch
import numpy as np
import hydra
import einops
import os
import sys
from pathlib import Path
from dynamics_modeling.eval import batch_eval_loop

sys.path.append(str(Path(__file__).parents[1]))

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

@hydra.main(config_path="config/", config_name="ode_two_grid.yaml")
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
    n_cond = cfg.data.n_cond

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    gamma_step = cfg.optim.gamma_step
    epochs = cfg.optim.epochs
    in_grid = cfg.optim.in_grid

    # inr
    load_run_name = cfg.inr.run_name
    load_run_name_suffix = cfg.inr.run_name_suffix
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
    grid_channel = cfg.inr.grid_channel
    grid_size = cfg.inr.grid_size

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

    # we need the latent dim and the sub_tr used for training
    if load_run_name is not None:
        multichannel = False
        tmp = torch.load(root_dir / "inr" / f"{load_run_name}.pt")
        latent_dim = tmp["cfg"].inr.latent_dim
        try:
            sub_from = tmp["cfg"].data.sub_from
        except:
            sub_from = 1

        sub_tr = tmp["cfg"].data.sub_tr
        seed = tmp["cfg"].data.seed

    elif load_run_dict is not None:
        multichannel = True
        tmp_data_to_encode = list(load_run_dict.keys())[0]
        tmp_run_name = list(load_run_dict.values())[0]
        tmp = torch.load(root_dir / tmp_data_to_encode /
                         "inr" / f"{tmp_run_name}.pt")
        latent_dim = tmp["cfg"].inr.latent_dim
        sub_tr = tmp["cfg"].data.sub_tr
        seed = tmp["cfg"].data.seed
        try:
            sub_from = tmp["cfg"].data.sub_from
        except:
            sub_from = 1

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
    print(f"data: {dataset_name}, u_train: {u_train.shape}, u_eval_extrapolation: {u_eval_extrapolation.shape}, u_test: {u_test.shape}, u_test_in: {u_test_in.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}, grid_te_in: {grid_te_in.shape}")
    print("same_grid : ", same_grid)
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
        u_eval_extrapolation, grid_tr_extra, latent_dim, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, latent_dim, dataset_name, data_to_encode
    )
   
    trainset_extra_in = TemporalDatasetWithCode(
        u_eval_extrapolation_in, grid_tr_extra_in, latent_dim, dataset_name, data_to_encode
    )
    testset_in = TemporalDatasetWithCode(
        u_test_in, grid_te_in, latent_dim, dataset_name, data_to_encode
    )

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

    if load_run_name is not None:
        inr, alpha = load_inr_model(
            root_dir / "inr",
            load_run_name+load_run_name_suffix,
            data_to_encode,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        print(inr)
        modulations = load_dynamics_modulations(
            trainset,
            trainset_extra,
            testset,
            inr,
            root_dir / "modulations",
            load_run_name,
            inner_steps=inner_steps,
            alpha=alpha,
            batch_size=2,
            data_to_encode=None,
            try_reload=False,
        )
        z_train = modulations["z_train"]
        z_train_extra = modulations["z_train_extra"]
        z_test = modulations["z_test"]

        modulations_in = load_dynamics_modulations(
            trainset,
            trainset_extra_in,
            testset_in,
            inr,
            root_dir / "modulations",
            load_run_name,
            inner_steps=inner_steps,
            alpha=alpha,
            batch_size=2,
            data_to_encode=None,
            try_reload=False,
        )
        z_train_extra_in = modulations_in["z_train_extra"]
        z_test_in = modulations_in["z_test"]
        # import pdb; pdb.set_trace()
        if cfg.dynamics.normalize_per_ele:

            z_mean = einops.rearrange(z_train, "b l t -> (b t) l").mean(0).reshape(1, latent_dim, 1)
            z_std = einops.rearrange(z_train, "b l t -> (b t) l").std(0).reshape(1, latent_dim, 1)
        elif cfg.dynamics.normalize_per_channel:
            z_train_rearrange = einops.rearrange(z_train, "b l t -> (b t) l")
            z_train_grid = z_train_rearrange.reshape(-1, grid_channel, grid_size*grid_size).permute(0,2,1).reshape(-1, grid_channel)
            z_mean = z_train_grid.mean(0).reshape(1, grid_channel, 1).repeat(1,1,grid_size*grid_size).reshape(1,-1,1)
            z_std = z_train_grid.std(0).reshape(1, grid_channel, 1).repeat(1,1,grid_size*grid_size).reshape(1,-1,1)
        else:
            z_mean = z_train.mean().reshape(1, 1, 1).repeat(1, latent_dim, 1)
            z_std = z_train.std().reshape(1, 1, 1).repeat(1, latent_dim, 1)
        # print(z_train.mean(), z_train.std())
        z_train = (z_train - z_mean) / z_std
        z_train_extra = (z_train_extra - z_mean) / z_std
        z_test = (z_test - z_mean) / z_std

        z_train_extra_in = (z_train_extra_in - z_mean) / z_std
        z_test_in = (z_test_in - z_mean) / z_std
        
        # visual 
        # from visualize import write_image  
        # visual_first = 4
        # visual_path=os.path.join(run.dir, 'traininter')
        # visual_mod = grid_size
        # modulations = modulations["z_train"]
        # modulations_v = modulations.permute(0,2,1)
        # modulations_v = modulations_v.reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
        # divider = 2 * modulations_v.shape[2]
        # modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
        # for visual_idx in range(visual_first):
        #     write_image(modulations_v[visual_idx], modulations_v[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)
        # import pdb; pdb.set_trace()

    elif load_run_dict is not None:
        inr_dict = {}
        z_mean = {}
        z_std = {}
        c = len(list(load_run_dict.keys()))
        z_train = torch.zeros(ntrain, latent_dim, c, T_train)
        z_train_extra = torch.zeros(ntrain, latent_dim, c, T_test)
        z_test = torch.zeros(ntest, latent_dim, c, T_test)

        for to_encode in list(load_run_dict.keys()):
            tmp_name = load_run_dict[to_encode]
            #print('tmp name', to_encode, tmp_name)
            output_dim = 1
            inr, alpha = load_inr_model(
                root_dir / to_encode / "inr",
                tmp_name,
                to_encode,
                input_dim=input_dim,
                output_dim=output_dim,
            )

            trainset.set_data_to_encode(to_encode)
            trainset_extra.set_data_to_encode(to_encode)
            testset.set_data_to_encode(to_encode)

            modulations = load_dynamics_modulations(
                trainset,
                trainset_extra,
                testset,
                inr,
                root_dir / to_encode / "modulations",
                tmp_name,
                inner_steps=inner_steps,
                alpha=alpha,
                batch_size=2,
                data_to_encode=to_encode,
                try_reload=False,
            )
            inr_dict[to_encode] = inr
            z_tr = modulations["z_train"]
            z_tr_extra = modulations["z_train_extra"]
            z_te = modulations["z_test"]
            z_m = einops.rearrange(z_tr, "b l t -> (b t) l").mean(0).reshape(1, latent_dim, 1) #.unsqueeze(0).unsqueeze(-1).repeat(1, 1, T)
            z_s = einops.rearrange(z_tr, "b l t -> (b t) l").std(0).reshape(1, latent_dim, 1) #.unsqueeze(0).unsqueeze(-1)
            z_mean[to_encode] = z_m
            z_std[to_encode] = z_s
            z_train[..., KEY_TO_INDEX[dataset_name]
                    [to_encode], :] = (z_tr - z_m) / z_s
            z_train_extra[..., KEY_TO_INDEX[dataset_name]
                    [to_encode], :] = (z_tr_extra - z_m) / z_s
            z_test[..., KEY_TO_INDEX[dataset_name]
                   [to_encode], :] = (z_te - z_m) / z_s

        # concat the code
        trainset.set_data_to_encode(None)
        trainset_extra.set_data_to_encode(None)
        testset.set_data_to_encode(None)
        # rename inr_dict <- inr
        inr = inr_dict

    trainset.z = z_train
    trainset_extra.z = z_train_extra
    testset.z = z_test
    trainset_extra_in.z = z_train_extra_in
    testset_in.z = z_test_in

    print('ztrain', z_train.shape, z_train.mean(), z_train.std())
    print('ztrain_extra', z_train_extra.shape, z_train_extra.mean(), z_train_extra.std())
    print('ztest', z_test.shape, z_test.mean(), z_test.std())
    print('ztrain_extra_in', z_train_extra_in.shape, z_train_extra_in.mean(), z_train_extra_in.std())
    print('ztest_in', z_test_in.shape, z_test_in.mean(), z_test_in.std())

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
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val//2,
        shuffle=False,
        num_workers=1,
    )

    train_extra_loader_in = torch.utils.data.DataLoader(
        trainset_extra_in,
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader_in = torch.utils.data.DataLoader(
        testset_in,
        batch_size=batch_size_val//2,
        shuffle=False,
        num_workers=1,
    )

    c = z_train.shape[2] if multichannel else 1 
    if model_type == 'ode':
        model = Derivative(c, z_train.shape[1], hidden, depth).cuda()
    elif model_type == 'fno':
        modes=cfg.dynamics.modes
        width=32
        model = DerivativeFNO(modes, modes, width, grid_channel, grid_size).cuda()
    else:
        raise NotImplementedError

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
        
    # eval 
    evaluate = cfg.wandb.evaluate
    if evaluate:
        pred_train_inter_mse, code_train_inter_mse, pred_train_extra_mse, code_train_extra_mse, total_pred_train_mse, detailed_train_eval_mse = batch_eval_loop(
                    model, inr, train_extra_loader,
                    timestamps_test, detailed_train_eval_mse,
                    ntrain, multichannel, z_mean, z_std,
                    dataset_name, T_train, n_cond, visual_first=4, visual_path=os.path.join(run.dir, f'train_{epoch_start}'), visual_mod=grid_size
                )

        pred_test_inter_mse, code_test_inter_mse, pred_test_extra_mse, code_test_extra_mse, pred_test_mse, detailed_test_mse = batch_eval_loop(
                    model, inr, test_loader,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel, z_mean, z_std,
                    dataset_name, T_train, n_cond, visual_first=4, visual_path=os.path.join(run.dir, f'test_{epoch_start}'), visual_mod=grid_size
                )
        print(f'train inter mse {pred_train_inter_mse}, train_extra_mse {pred_train_extra_mse}, test inter mse {pred_test_inter_mse}, test extra mse {pred_test_extra_mse}')
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
            modulations = modulations.cuda()
            coords = coords.cuda()
            n_samples = images.shape[0]

            if multichannel:
                modulations = einops.rearrange(modulations, "b l c t -> b (l c) t")

            modulations = modulations[...,n_cond:]
            images = images[...,n_cond:]
            coords = coords[...,n_cond:]

            z_pred = ode_scheduling(odeint, model, modulations, timestamps_train[n_cond:], epsilon_t)
            loss = ((z_pred - modulations) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            if True in (step_show, step_show_last):

                pred = get_reconstructions(
                    inr, coords, z_pred, z_mean, z_std, dataset_name
                )
                pred_train_mse += ((pred - images) ** 2).mean() * n_samples
                
                if multichannel:
                    detailed_train_mse.aggregate(pred, images)

        code_train_mse = code_train_mse / ntrain

        if True in (step_show, step_show_last):
            pred_train_mse = pred_train_mse / ntrain

        scheduler.step(code_train_mse)

        if T_train != T_test:
            pred_train_inter_mse, code_train_inter_mse, pred_train_extra_mse, code_train_extra_mse, total_pred_train_mse, detailed_train_eval_mse = batch_eval_loop(
                model, inr, train_extra_loader,
                timestamps_test, detailed_train_eval_mse,
                ntrain, multichannel, z_mean, z_std,
                dataset_name, T_train, n_cond
            )

            pred_train_inter_mse_in, code_train_inter_mse_in, pred_train_extra_mse_in, code_train_extra_mse_in, total_pred_train_mse_in, detailed_train_eval_mse = batch_eval_loop(
                model, inr, train_extra_loader_in,
                timestamps_test, detailed_train_eval_mse,
                ntrain, multichannel, z_mean, z_std,
                dataset_name, T_train, n_cond
            )

        if True in (step_show, step_show_last):
            if T_train != T_test:
                pred_test_inter_mse, code_test_inter_mse, pred_test_extra_mse, code_test_extra_mse, pred_test_mse, detailed_test_mse = batch_eval_loop(
                    model, inr, test_loader,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel, z_mean, z_std,
                    dataset_name, T_train,n_cond
                )

                pred_test_inter_mse_in, code_test_inter_mse_in, pred_test_extra_mse_in, code_test_extra_mse_in, pred_test_mse_in, detailed_test_mse = batch_eval_loop(
                    model, inr, test_loader_in,
                    timestamps_test, detailed_test_mse,
                    ntest, multichannel, z_mean, z_std,
                    dataset_name, T_train,n_cond
                )
                print(f'{step} code_train_inter_mse {code_train_inter_mse:2.e}, code_train_extra_mse {code_train_extra_mse:2.e}, pred_train_inter_mse {pred_train_inter_mse:2.e}, pred_train_extra_mse {pred_train_extra_mse:2.e}')
                print(f'{step} code_train_inter_mse_in {code_train_inter_mse_in:2.e}, code_train_extra_mse_in {code_train_extra_mse_in:2.e}, pred_train_inter_mse_in {pred_train_inter_mse_in:2.e}, pred_train_extra_mse_in {pred_train_extra_mse_in:2.e}')
                print(f'{step} code_test_inter_mse {code_test_inter_mse:2.e}, code_test_extra_mse {code_test_extra_mse:2.e}, pred_test_inter_mse {pred_test_inter_mse:2.e}, pred_test_extra_mse {pred_test_extra_mse:2.e}')
                print(f'{step} code_test_inter_mse_in {code_test_inter_mse_in:2.e}, code_test_extra_mse_in {code_test_extra_mse_in:2.e}, pred_test_inter_mse_in {pred_test_inter_mse_in:2.e}, pred_test_extra_mse_in {pred_test_extra_mse_in:2.e}')
                
                log_dic = {
                    "pred_train_inter_mse": pred_train_inter_mse,
                    "pred_train_extra_mse": pred_train_extra_mse,
                    "pred_test_mse_inter": pred_test_inter_mse,
                    "pred_test_mse_extra": pred_test_extra_mse,
                    # "pred_test_mse": pred_test_mse,
                    "code_train_inter_mse": code_train_inter_mse,
                    "code_train_extra_mse": code_train_extra_mse,
                    "code_test_inter_mse": code_test_inter_mse,
                    "code_test_extra_mse": code_test_extra_mse,
                }
                
                if multichannel:

                    dic_train_mse = detailed_train_mse.get_dic()
                    detailed_train_mse.reset_dic()

                    dic_train_extra_mse = detailed_train_eval_mse.get_dic()
                    detailed_train_eval_mse.reset_dic()

                    dic_test_mse = detailed_test_mse.get_dic()
                    detailed_test_mse.reset_dic()

                    log_dic.update(dic_train_mse)
                    log_dic.update(dic_train_extra_mse)
                    log_dic.update(dic_test_mse)

            elif T_train == T_test:
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
                    "pred_train_mse": pred_train_mse,
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
                        "alpha": alpha,
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
                        "loss": code_test_mse,
                        "alpha": alpha,
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
                    "alpha": alpha,
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
                    "loss": code_test_mse,
                    "alpha": alpha,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{model_dir}/{run_name}_ck.pt",
            )
    return total_pred_train_mse
    
if __name__ == "__main__":
    main()