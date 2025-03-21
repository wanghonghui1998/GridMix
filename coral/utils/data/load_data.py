import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
# import xarray as xr
from scipy import io
from torch.utils.data import Dataset

# MP-PDE imports
# from equations.PDEs import CE
# from common.utils import HDF5Dataset
from coral.utils.data.setting import init_setting

# if dataset_name == "shallow-water":
#    index = 0 if data_to_encode == "height" else 1
#    x_train = x_train[..., index].unsqueeze(-1)
#    y_train = y_train[..., index].unsqueeze(-1)

#    x_test = x_test[..., index].unsqueeze(-1)
#    y_test = y_test[..., index].unsqueeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_operator_data(
    data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.
    
    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.

    Raises:
        NotImplementedError: _description_

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
        grid_tr (torch.Tensor): coordinates of x_train, y_train
        grid_te (torch.Tensor): coordinates of x_test, y_test
    """

    data_dir = Path(data_dir)

    if dataset_name == "burgers":
        min_sub = 8  # 4 -> 2048  # 1024
        x_train, y_train, x_test, y_test = get_burgers(
            data_dir / "burgers_data_R10.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "darcy":
        min_sub = 1
        x_train, y_train = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth1.mat", ntrain
        )
        x_test, y_test = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth2.mat", ntest
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

        # x_max, x_min = x_train.max(), x_train.min()
        # x_train, x_test = (
        #    (x_train - x_min) / (x_max - x_min),
        #    (x_test - x_min) / (x_max - x_min),
        # )
        mu, sigma = x_train.mean(), x_train.std()
        x_train, x_test = (x_train - mu) / sigma, (x_test - mu) / sigma

    elif dataset_name == "kdv":
        min_sub = 8  # 8 -> 1024
        x_train, y_train, x_test, y_test = get_kdv(
            data_dir / "kdv_train_test.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "navier-stokes":
        min_sub = 1
        reader = MatReader(data_dir / "ns_V1e-3_N5000_T50.mat")
        u = reader.read_field("u")

        u_train = u[:ntrain, :]
        u_test = u[-ntest:, :]

        x_train, y_train = u_train[..., 9], u_train[..., 19]  # 19 previously
        x_test, y_test = u_test[..., 9], u_test[..., 19]  # 19 previously

    elif dataset_name == "navier-stokes-256":
        min_sub = 2  # 2 usually
        # train_dir = str(data_dir) + "/" + "navier_1e-3_256_2_train.shelve"
        # test_dir = str(data_dir) + "/" + "navier_1e-3_256_2_test.shelve"

        # data_train = dict(shelve.open(train_dir))
        # data_test = dict(shelve.open(test_dir))

        shelve_dir = str(data_dir) + "/" + "ns_1e-3_1200.shelve"
        data = dict(shelve.open(shelve_dir))

        data.pop("a")
        data.pop("t")

        # concatenate dictionary to be of shape (ntrain, 40, 256, 256)

        u = torch.tensor(
            np.concatenate(
                list(
                    map(lambda key: np.expand_dims(
                        np.array(data[key]), 0), data.keys())
                )
            )
        )

        u_train = u[:1000]
        u_test = u[-200:]

        # select the 9 and 19 indices

        x_train, y_train = u_train[..., 9], u_train[..., 19]
        x_test, y_test = u_test[..., 9], u_test[..., 19]

        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    elif dataset_name == "advection":
        min_sub = 1
        x_train, y_train = get_advection(data_dir / "train_IC2.npz", ntrain)
        x_test, y_test = get_advection(data_dir / "test_IC2.npz", ntest)

    elif dataset_name == "pipe":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_pipe(
            data_dir, ntrain=1000, ntest=200, min_sub=min_sub
        )

    elif dataset_name == "airfoil":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_airfoil(
            data_dir, ntrain=1000, ntest=200, min_sub=min_sub
        )

    elif dataset_name == "elasticity":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_elasticity(
            data_dir, ntrain=1000, ntest=200, min_sub=min_sub
        )

    elif dataset_name == "shallow-water":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_shallow_water(
            data_dir, ntrain, ntest, min_sub=1
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[..., ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    else:
        raise NotImplementedError

    # expects x_train, y_train, x_test, y_test to be of shape (N, dx, C) or (N, dx1, dx2, C)
    # expects grid_inp_tr, grid_out_tr, x_test, y_test to be of shape (N, dx, C) or (N, dx1, dx2, C)

    if dataset_name in ["elasticity"]:
        average_grid = x_train.mean(0)
        grid_tr = average_grid
        grid_te = average_grid

        grid_tr = einops.repeat(grid_tr, '... -> b ...', b=x_train.shape[0])
        grid_te = einops.repeat(grid_te, '... -> b ...', b=x_test.shape[0])

    elif dataset_name in ["shallow-water"]:
        grid_tr = shape2spherical_coordinates(x_train.shape[1:-1])
        grid_te = shape2spherical_coordinates(x_test.shape[1:-1])
    
    elif dataset_name in ['airfoil', 'pipe']:
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

        # create grid for the high resolution data
        grid_tr = shape2coordinates(x_train.shape[1:-1])
        grid_te = shape2coordinates(x_test.shape[1:-1])

        #if dataset_name == "pipe":
            #grid_tr[..., 0] = grid_tr[..., 0]*2 # the other way around
            #grid_te[..., 0] = grid_te[..., 0]*2
            #grid_tr[..., 1] = grid_tr[..., 1]*0.2
            #grid_te[..., 1] = grid_te[..., 1]*0.2
            #grid_tr[..., 1] = grid_tr[..., 0]*10
            #grid_te[..., 1] = grid_te[..., 0]*10

        grid_tr = einops.repeat(grid_tr, '... -> b ...', b=x_train.shape[0])
        grid_te = einops.repeat(grid_te, '... -> b ...', b=x_test.shape[0])

    else:
        x_train = x_train.unsqueeze(-1)
        x_test = x_test.unsqueeze(-1)

        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)        

        # create grid for the high resolution data
        grid_tr = shape2coordinates(x_train.shape[1:-1])
        grid_te = shape2coordinates(x_test.shape[1:-1])

    # subample data and grid, either uniformly (sub is int) or not (sub is float)

    if isinstance(sub_tr, int):
        grid_tr = operator_subsample(grid_tr, sub_tr)
        x_train = operator_subsample(x_train, sub_tr)
        y_train = operator_subsample(y_train, sub_tr)

    if isinstance(sub_te, int):
        grid_te = operator_subsample(grid_te, sub_te)
        x_test = operator_subsample(x_test, sub_te)
        y_test = operator_subsample(y_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            N = x_train.shape[0]
            C = x_train.shape[-1]
            perm = torch.randperm(x_train.reshape(N, -1, C).shape[1])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]

            grid_tr = operator_subsample(grid_tr, mask_tr)
            x_train = operator_subsample(x_train, mask_tr)
            y_train = operator_subsample(y_train, mask_tr)
        else:
            x_train, y_train, grid_tr, perm = operator_different_subsample(
                x_train, y_train, grid_tr, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            N = x_test.shape[0]
            C = x_test.shape[-1]
            perm = torch.randperm(x_test.reshape(N, -1, C).shape[1])
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]

            grid_te = operator_subsample(grid_te, mask_te)
            x_test = operator_subsample(x_test, mask_te)
            y_test = operator_subsample(y_test, mask_te)
        else:
            x_test, y_test, grid_te, perm = operator_different_subsample(
                x_test, y_test, grid_te, sub_te
            )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        grid_tr,
        grid_te,
    )

def get_dynamics_data_two_grid(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        index_start = 9
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat", 1000, 200, sequence_length, index_start
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(data_dir, seq_inter_len, seq_extra_len)
    
    elif dataset_name == "navier-stokes-dino-first40-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_first40_T100(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-mid40-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_mid40_T100(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-dino-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_T100(data_dir, seq_inter_len, seq_extra_len)
    
    elif dataset_name == "navier-stokes-dino-first40-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_first40_T100_2048(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-mid40-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_mid40_T100_2048(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-dino-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_T100_2048(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_last40_total64(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-f40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-f40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-60":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total60(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-40-70":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total70(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "sst-11-22":
        u_train, u_eval_extrapolation, u_test = get_sst_11_22(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "KS-fixed-viscosity-fixed-region":
        u_train, u_eval_extrapolation, u_test = get_KS_fixed_viscosity_fixed_region(data_dir)
    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])
    elif dataset_name == "KS-fixed-viscosity-fixed-region":
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )

    # import pdb; pdb.set_trace()
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    # if isinstance(sub_from, int):
    #     grid_tr = dynamics_subsample(grid_tr, sub_from)
    #     u_train = dynamics_subsample(u_train, sub_from)
    # grid_tr_extra_mask_tr = grid_tr_extra.clone()
    # u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone()
    # grid_te_mask_tr = grid_te.clone()
    # u_test_mask_tr = u_test.clone()
    # import pdb; pdb.set_trace()
    
    grid_tr_full = grid_tr.clone() 
    u_train_full = u_train.clone()
    grid_tr_extra_full = grid_tr_extra.clone()
    u_eval_extrapolation_full = u_eval_extrapolation.clone()
    grid_te_full = grid_te.clone()
    u_test_full = u_test.clone() 
    
    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr_full, sub_tr)
        u_train = dynamics_subsample(u_train_full, sub_tr)

        grid_tr_extra_in = dynamics_subsample(grid_tr_extra_full, sub_tr)
        u_eval_extrapolation_in = dynamics_subsample(u_eval_extrapolation_full, sub_tr)

        grid_te_in = dynamics_subsample(grid_te_full, sub_tr)
        u_test_in = dynamics_subsample(u_test_full, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra_full, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation_full, sub_tr)

        grid_tr_out1 = dynamics_subsample(grid_tr_full, sub_tr)
        u_train_out1 = dynamics_subsample(u_train_full, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te_full, sub_te)
        u_test = dynamics_subsample(u_test_full, sub_te)

        grid_tr_out2 = dynamics_subsample(grid_tr_full, sub_te)
        u_train_out2 = dynamics_subsample(u_train_full, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            # import pdb; pdb.set_trace()
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            # torch.manual_seed(0)
            perm = torch.randperm(num_points)
            # print(perm[:10])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr_full, mask_tr)
            u_train = dynamics_subsample(u_train_full, mask_tr)

            grid_tr_extra_in = dynamics_subsample(grid_tr_extra_full, mask_tr)
            u_eval_extrapolation_in = dynamics_subsample(u_eval_extrapolation_full, mask_tr)

            grid_te_in = dynamics_subsample(grid_te_full, mask_tr)
            u_test_in = dynamics_subsample(u_test_full, mask_tr)

        else:
            print("computing different grids")
            # u_train, grid_tr, perm = dynamics_different_subsample(
            #     u_train, grid_tr, sub_tr
            # )
            u_train, grid_tr, u_eval_extrapolation_in, grid_tr_extra_in, u_test_in, grid_te_in, perm = dynamics_different_subsample_two_grid(
                u_train_full, grid_tr_full, sub_tr, u_eval_extrapolation_full, grid_tr_extra_full, u_test_full, grid_te_full
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            # debug
            # grid_tr_extra_mask_tr = dynamics_subsample(grid_tr_extra, mask_tr)
            # u_eval_extrapolation_mask_tr = dynamics_subsample(u_eval_extrapolation, mask_tr)

            grid_tr_extra = dynamics_subsample(grid_tr_extra_full, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation_full, mask_tr_eval)

            grid_tr_out1 = dynamics_subsample(grid_tr_full, mask_tr_eval)
            u_train_out1 = dynamics_subsample(u_train_full, mask_tr_eval)

        else:
            # u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
            #     u_eval_extrapolation, grid_tr_extra, sub_tr
            # )
            u_eval_extrapolation, grid_tr_extra, u_train_out1, grid_tr_out1, _, _, perm = dynamics_different_subsample_two_grid(
                u_eval_extrapolation_full, grid_tr_extra_full, sub_tr, u_train_full, grid_tr_full, None, None
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            # debug
            # grid_te_mask_tr = dynamics_subsample(grid_te, mask_tr)
            # u_test_mask_tr = dynamics_subsample(u_test, mask_tr)

            grid_te = dynamics_subsample(grid_te_full, mask_te)
            u_test = dynamics_subsample(u_test_full, mask_te)

            grid_tr_out2 = dynamics_subsample(grid_tr_full, mask_te)
            u_train_out2 = dynamics_subsample(u_train_full, mask_te)

        else:
            # u_test, grid_te, perm = dynamics_different_subsample(
            #     u_test, grid_te, sub_te
            # )
            u_test, grid_te, u_train_out2, grid_tr_out2, _, _, perm = dynamics_different_subsample_two_grid(
                u_test_full, grid_te_full, sub_te, u_train_full, grid_tr_full, None, None
            )

    return u_train, u_train_out1, u_train_out2, u_eval_extrapolation, u_eval_extrapolation_in, u_test, u_test_in, grid_tr, grid_tr_out1, grid_tr_out2, grid_tr_extra, grid_tr_extra_in, grid_te, grid_te_in
    # return u_train, u_eval_extrapolation, u_test, u_eval_extrapolation_mask_tr, u_test_mask_tr, grid_tr, grid_tr_extra, grid_te, grid_tr_extra_mask_tr, grid_te_mask_tr

def get_dynamics_data_three_grid(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        index_start = 9
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat", 1000, 200, sequence_length, index_start
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_last40_total64(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-f40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-f40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-60":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total60(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-40-70":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total70(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "sst-11-22":
        u_train, u_eval_extrapolation, u_test = get_sst_11_22(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len)

    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )

    # import pdb; pdb.set_trace()
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    # if isinstance(sub_from, int):
    #     grid_tr = dynamics_subsample(grid_tr, sub_from)
    #     u_train = dynamics_subsample(u_train, sub_from)
    # grid_tr_extra_mask_tr = grid_tr_extra.clone()
    # u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone()
    # grid_te_mask_tr = grid_te.clone()
    # u_test_mask_tr = u_test.clone()
    # import pdb; pdb.set_trace()
    
    grid_tr_full = grid_tr.clone() 
    u_train_full = u_train.clone()
    grid_tr_extra_full = grid_tr_extra.clone()
    u_eval_extrapolation_full = u_eval_extrapolation.clone()
    grid_te_full = grid_te.clone()
    u_test_full = u_test.clone() 
    
    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr_full, sub_tr)
        u_train = dynamics_subsample(u_train_full, sub_tr)

        grid_tr_extra_in = dynamics_subsample(grid_tr_extra_full, sub_tr)
        u_eval_extrapolation_in = dynamics_subsample(u_eval_extrapolation_full, sub_tr)

        grid_te_in = dynamics_subsample(grid_te_full, sub_tr)
        u_test_in = dynamics_subsample(u_test_full, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra_full, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation_full, sub_tr)

        grid_tr_out1 = dynamics_subsample(grid_tr_full, sub_tr)
        u_train_out1 = dynamics_subsample(u_train_full, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te_full, sub_te)
        u_test = dynamics_subsample(u_test_full, sub_te)

        grid_tr_out2 = dynamics_subsample(grid_tr_full, sub_te)
        u_train_out2 = dynamics_subsample(u_train_full, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            # import pdb; pdb.set_trace()
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            # torch.manual_seed(0)
            perm = torch.randperm(num_points)
            # print(perm[:10])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr_full, mask_tr)
            u_train = dynamics_subsample(u_train_full, mask_tr)

            grid_tr_extra_in = dynamics_subsample(grid_tr_extra_full, mask_tr)
            u_eval_extrapolation_in = dynamics_subsample(u_eval_extrapolation_full, mask_tr)

            grid_te_in = dynamics_subsample(grid_te_full, mask_tr)
            u_test_in = dynamics_subsample(u_test_full, mask_tr)

        else:
            print("computing different grids")
            # u_train, grid_tr, perm = dynamics_different_subsample(
            #     u_train, grid_tr, sub_tr
            # )
            u_train, grid_tr, u_eval_extrapolation_in, grid_tr_extra_in, u_test_in, grid_te_in, perm = dynamics_different_subsample_two_grid(
                u_train_full, grid_tr_full, sub_tr, u_eval_extrapolation_full, grid_tr_extra_full, u_test_full, grid_te_full
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            # debug
            # grid_tr_extra_mask_tr = dynamics_subsample(grid_tr_extra, mask_tr)
            # u_eval_extrapolation_mask_tr = dynamics_subsample(u_eval_extrapolation, mask_tr)

            grid_tr_extra = dynamics_subsample(grid_tr_extra_full, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation_full, mask_tr_eval)

            grid_tr_out1 = dynamics_subsample(grid_tr_full, mask_tr_eval)
            u_train_out1 = dynamics_subsample(u_train_full, mask_tr_eval)

        else:
            # u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
            #     u_eval_extrapolation, grid_tr_extra, sub_tr
            # )
            u_eval_extrapolation, grid_tr_extra, u_train_out1, grid_tr_out1, _, _, perm = dynamics_different_subsample_two_grid(
                u_eval_extrapolation_full, grid_tr_extra_full, sub_tr, u_train_full, grid_tr_full, None, None
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            # debug
            # grid_te_mask_tr = dynamics_subsample(grid_te, mask_tr)
            # u_test_mask_tr = dynamics_subsample(u_test, mask_tr)

            grid_te = dynamics_subsample(grid_te_full, mask_te)
            u_test = dynamics_subsample(u_test_full, mask_te)

            grid_tr_out2 = dynamics_subsample(grid_tr_full, mask_te)
            u_train_out2 = dynamics_subsample(u_train_full, mask_te)

        else:
            # u_test, grid_te, perm = dynamics_different_subsample(
            #     u_test, grid_te, sub_te
            # )
            u_test, grid_te, u_train_out2, grid_tr_out2, _, _, perm = dynamics_different_subsample_two_grid(
                u_test_full, grid_te_full, sub_te, u_train_full, grid_tr_full, None, None
            )

    
    return u_train, u_train_out1, u_train_out2, u_eval_extrapolation, u_eval_extrapolation_in, u_test, u_test_in, grid_tr, grid_tr_out1, grid_tr_out2, grid_tr_extra, grid_tr_extra_in, grid_te, grid_te_in, u_train_full, grid_tr_full, u_eval_extrapolation_full, grid_tr_extra_full, u_test_full, grid_te_full
    # return u_train, u_eval_extrapolation, u_test, u_eval_extrapolation_mask_tr, u_test_mask_tr, grid_tr, grid_tr_extra, grid_te, grid_tr_extra_mask_tr, grid_te_mask_tr


def get_dynamics_data(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        index_start = 9
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat", 1000, 200, sequence_length, index_start
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-dino-first40-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_first40_T100(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-mid40-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_mid40_T100(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-dino-T100":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_T100(data_dir, seq_inter_len, seq_extra_len)
    
    elif dataset_name == "navier-stokes-dino-first40-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_first40_T100_2048(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-mid40-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_mid40_T100_2048(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-dino-T100-2048":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_T100_2048(data_dir, seq_inter_len, seq_extra_len)


    elif dataset_name == "navier-stokes-dino-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_last40_total64(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-f40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-f40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-60":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total60(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-40-70":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total70(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "sst-11-22":
        u_train, u_eval_extrapolation, u_test = get_sst_11_22(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "KS-fixed-viscosity-fixed-region":
        u_train, u_eval_extrapolation, u_test = get_KS_fixed_viscosity_fixed_region(data_dir)
    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    elif dataset_name == "KS-fixed-viscosity-fixed-region":
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )
    # import pdb; pdb.set_trace()
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    # if isinstance(sub_from, int):
    #     grid_tr = dynamics_subsample(grid_tr, sub_from)
    #     u_train = dynamics_subsample(u_train, sub_from)
    # grid_tr_extra_mask_tr = grid_tr_extra.clone()
    # u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone()
    # grid_te_mask_tr = grid_te.clone()
    # u_test_mask_tr = u_test.clone()
    # import pdb; pdb.set_trace()
    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr, sub_tr)
        u_train = dynamics_subsample(u_train, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te, sub_te)
        u_test = dynamics_subsample(u_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            # import pdb; pdb.set_trace()
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            # torch.manual_seed(0)
            perm = torch.randperm(num_points)
            # print(perm[:10])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr, mask_tr)
            u_train = dynamics_subsample(u_train, mask_tr)

        else:
            print("computing different grids")
            u_train, grid_tr, perm = dynamics_different_subsample(
                u_train, grid_tr, sub_tr
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            # debug
            # grid_tr_extra_mask_tr = dynamics_subsample(grid_tr_extra, mask_tr)
            # u_eval_extrapolation_mask_tr = dynamics_subsample(u_eval_extrapolation, mask_tr)

            grid_tr_extra = dynamics_subsample(grid_tr_extra, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, mask_tr_eval)

        else:
            u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
                u_eval_extrapolation, grid_tr_extra, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            # debug
            # grid_te_mask_tr = dynamics_subsample(grid_te, mask_tr)
            # u_test_mask_tr = dynamics_subsample(u_test, mask_tr)

            grid_te = dynamics_subsample(grid_te, mask_te)
            u_test = dynamics_subsample(u_test, mask_te)

        else:
            u_test, grid_te, perm = dynamics_different_subsample(
                u_test, grid_te, sub_te
            )

    return u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te
    # return u_train, u_eval_extrapolation, u_test, u_eval_extrapolation_mask_tr, u_test_mask_tr, grid_tr, grid_tr_extra, grid_te, grid_tr_extra_mask_tr, grid_te_mask_tr


def get_dynamics_data_with_full_data(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        index_start = 9
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat", 1000, 200, sequence_length, index_start
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_last40_total64(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-f40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-f40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-60":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total60(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-40-70":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total70(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "sst-11-22":
        u_train, u_eval_extrapolation, u_test = get_sst_11_22(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len)

    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )
    # import pdb; pdb.set_trace()
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)
    full_grid_te = grid_te.clone()
    full_u_test = u_test.clone()
    # if isinstance(sub_from, int):
    #     grid_tr = dynamics_subsample(grid_tr, sub_from)
    #     u_train = dynamics_subsample(u_train, sub_from)
    # grid_tr_extra_mask_tr = grid_tr_extra.clone()
    # u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone()
    # grid_te_mask_tr = grid_te.clone()
    # u_test_mask_tr = u_test.clone()
    # import pdb; pdb.set_trace()
    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr, sub_tr)
        u_train = dynamics_subsample(u_train, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te, sub_te)
        u_test = dynamics_subsample(u_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            # import pdb; pdb.set_trace()
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            # torch.manual_seed(0)
            perm = torch.randperm(num_points)
            # print(perm[:10])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr, mask_tr)
            u_train = dynamics_subsample(u_train, mask_tr)

        else:
            print("computing different grids")
            u_train, grid_tr, perm = dynamics_different_subsample(
                u_train, grid_tr, sub_tr
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            # debug
            # grid_tr_extra_mask_tr = dynamics_subsample(grid_tr_extra, mask_tr)
            # u_eval_extrapolation_mask_tr = dynamics_subsample(u_eval_extrapolation, mask_tr)

            grid_tr_extra = dynamics_subsample(grid_tr_extra, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, mask_tr_eval)

        else:
            u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
                u_eval_extrapolation, grid_tr_extra, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            # debug
            # grid_te_mask_tr = dynamics_subsample(grid_te, mask_tr)
            # u_test_mask_tr = dynamics_subsample(u_test, mask_tr)

            grid_te = dynamics_subsample(grid_te, mask_te)
            u_test = dynamics_subsample(u_test, mask_te)

        else:
            u_test, grid_te, perm = dynamics_different_subsample(
                u_test, grid_te, sub_te
            )
# full_grid_te = grid_te.clone()
#     full_u_test = u_test.clone()
    return u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te, full_u_test, full_grid_te

def get_dynamics_data_extra_grid(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        index_start = 9
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat", 1000, 200, sequence_length, index_start
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-dino-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino_last40_total64(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-f40-64-wonorm":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64_wonorm(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-f40-64":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_first40_total64(data_dir, seq_inter_len, seq_extra_len)
    elif dataset_name == "navier-stokes-nms-40-60":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total60(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "navier-stokes-nms-40-70":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_nms_last40_total70(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "sst-11-22":
        u_train, u_eval_extrapolation, u_test = get_sst_11_22(data_dir, seq_inter_len, seq_extra_len)

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len)

    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )
    # import pdb; pdb.set_trace()
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    # if isinstance(sub_from, int):
    #     grid_tr = dynamics_subsample(grid_tr, sub_from)
    #     u_train = dynamics_subsample(u_train, sub_from)
    # grid_tr_extra_mask_tr = grid_tr_extra.clone()
    # u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone()
    # grid_te_mask_tr = grid_te.clone()
    # u_test_mask_tr = u_test.clone()
    # import pdb; pdb.set_trace()
    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr, sub_tr)
        u_train = dynamics_subsample(u_train, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te, sub_te)
        u_test = dynamics_subsample(u_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            # import pdb; pdb.set_trace()
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            # torch.manual_seed(0)
            perm = torch.randperm(num_points)
            # print(perm[:10])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr, mask_tr)
            u_train = dynamics_subsample(u_train, mask_tr)

        else:
            print("computing different grids")
            u_train, grid_tr, perm = dynamics_different_subsample(
                u_train, grid_tr, sub_tr
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            # debug
            # grid_tr_extra_mask_tr = dynamics_subsample(grid_tr_extra, mask_tr)
            # u_eval_extrapolation_mask_tr = dynamics_subsample(u_eval_extrapolation, mask_tr)

            grid_tr_extra = dynamics_subsample(grid_tr_extra, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, mask_tr_eval)
            grid_tr_extra_mask_tr = grid_tr_extra.clone()
            u_eval_extrapolation_mask_tr = u_eval_extrapolation.clone() 

        else:
            u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
                u_eval_extrapolation, grid_tr_extra, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            # debug
            # grid_te_mask_tr = dynamics_subsample(grid_te, mask_tr)
            # u_test_mask_tr = dynamics_subsample(u_test, mask_tr)

            grid_te = dynamics_subsample(grid_te, mask_te)
            u_test = dynamics_subsample(u_test, mask_te)

            grid_te_mask_tr = grid_te.clone()
            u_test_mask_tr = u_test.clone() 

        else:
            u_test, grid_te, perm = dynamics_different_subsample(
                u_test, grid_te, sub_te
            )

    # return u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te
    return u_train, u_eval_extrapolation, u_test, u_eval_extrapolation_mask_tr, u_test_mask_tr, grid_tr, grid_tr_extra, grid_te, grid_tr_extra_mask_tr, grid_te_mask_tr


def get_kdv(filename, ntrain, ntest):
    """Get kdv data.

    Args:
        filename (str or Path): path to dataset
        ntrain (int): number of training samples
        ntest (int): number of test samples

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
    """
    rw_ = io.loadmat(filename)
    x_data = rw_["input"].astype(np.float32)
    y_data = rw_["output"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_advection(filename, ndata):
    """Get advection data.

    Args:
        filename (str or Path): path to dataset

    Returns:
        x (torch.Tensor)
        y (torch.Tensor)
    """
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # (N, nt, nx)
    N = u.shape[0]

    x = u[:, 0, :].copy()  # (N, nx)
    y = u[:, 1:, :].copy()  # (N, (nt-1), nx)

    x = x[:ndata, :]  # (ndata, nx)
    y = y[:ndata, :, :]  # (ndata, (nt-1), nx)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y


def get_darcy_pwc(filename, ndata):
    """Get darcy data.

    Args:
        filename (str or Path): path to dataset
        ndata (int): number of samples to return

    Returns:
        x (torch.Tensor): (ndata, ..., 1)
        y (torch.Tensor): (ndata, ..., 1)
    """
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, :, :].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, :, :].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    x_branch = torch.from_numpy(x_branch)
    y = torch.from_numpy(y)

    return x_branch, y


def get_burgers(filename, ntrain, ntest):
    """Get burgers data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat(filename)
    x_data = data["a"].astype(np.float32)
    y_data = data["u"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :].astype(np.float32)  # added
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :].astype(np.float32)  # added

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_pipe(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    INPUT_X = os.path.join(filename, "pipe/Pipe_X.npy")
    INPUT_Y = os.path.join(filename, "pipe/Pipe_Y.npy")
    OUTPUT_Sigma = os.path.join(filename, "pipe/Pipe_Q.npy")

    N = ntrain + ntest
    r1 = min_sub
    r2 = min_sub
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    # input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 0]
    output = torch.tensor(output, dtype=torch.float)

     # new
    #xmax = torch.max(inputX[:ntrain])
    #xmin = torch.min(inputX[:ntrain])
    #inputX = inputX / 10
    #inputX = (inputX - xmin) / (xmax - xmin)

    #ymax = torch.max(inputY[:ntrain])
    #ymin = torch.min(inputY[:ntrain])
    #inputY = inputY / 10
    #inputY = (inputY - ymin) / (ymax - ymin)
    
    # new
    xmax = torch.max(inputX[:ntrain])
    xmin = torch.min(inputX[:ntrain])
    inputX = (inputX - xmin) / (xmax - xmin)

    ymax = torch.max(inputY[:ntrain])
    ymin = torch.min(inputY[:ntrain])
    inputY = (inputY - ymin) / (ymax - ymin)

    x = torch.cat([inputX.unsqueeze(-1), inputY.unsqueeze(-1)], axis=-1)
    x_train = x[:N][:ntrain, ::r1, ::r2, :][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = x[:N][-ntest:, ::r1, ::r2, :][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]

    # old

    #x_train = inputY[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    #y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    #x_test = inputY[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    #y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    # x_train = x_train.reshape(ntrain, s1, s2, 2)
    # x_test = x_test.reshape(ntest, s1, s2, 2)

    #xmax = torch.max(x_train)
    #xmin = torch.min(x_train)
    #x_train = (x_train - xmin) / (xmax - xmin)
    #x_test = (x_test - xmin) / (xmax - xmin)

    return x_train, y_train, x_test, y_test


def get_airfoil(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    INPUT_X = os.path.join(filename, "airfoil/naca/NACA_Cylinder_X.npy")
    INPUT_Y = os.path.join(filename, "airfoil/naca/NACA_Cylinder_Y.npy")
    OUTPUT_Sigma = os.path.join(
        filename, "airfoil/naca/NACA_Cylinder_Q.npy")

    r1 = min_sub
    r2 = min_sub
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    # input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)

    print('inputX', inputX.shape, inputX.mean())
    print('inputY', inputY.shape, inputY.mean())
    print('output_Sigma', output.shape, output.mean())
    print('minsub', r1, r2, s1, s2)

    # changed
    #x_train = inputY[:ntrain, ::r1, ::r2][:, :s1, :s2]
    #y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    #x_test = inputY[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2] # changed
    #y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]

    # new
    xmax = torch.max(inputX[:ntrain])
    xmin = torch.min(inputX[:ntrain])
    inputX = (inputX - xmin) / (xmax - xmin)

    ymax = torch.max(inputY[:ntrain])
    ymin = torch.min(inputY[:ntrain])
    inputY = (inputY - ymin) / (ymax - ymin)

    print('xmax', xmax, 'xmin', xmin)
    print('ymax', ymax, 'ymin', ymin)

    x = torch.cat([inputX.unsqueeze(-1), inputY.unsqueeze(-1)], axis=-1)
    x_train = x[:ntrain, ::r1, ::r2, :][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = x[ntrain:ntrain+ntest, ::r1, ::r2, :][:, :s1, :s2]
    y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]

    print(x_train.shape, x_train.min(), x_train.max(), x_train.reshape(-1, 2).max(0), x_train.reshape(-1, 2).min(0))

    return x_train, y_train, x_test, y_test


def get_elasticity(filename, ntrain, ntest, min_sub=1):
    PATH_Sigma = os.path.join(
        filename, "elasticity/Meshes/Random_UnitCell_sigma_10.npy")
    PATH_XY = os.path.join(filename, "elasticity/Meshes/Random_UnitCell_XY_10.npy")
    PATH_rr = os.path.join(filename, "elasticity/Meshes/Random_UnitCell_rr_10.npy")
    PATH_theta = os.path.join(
        filename, "elasticity/Meshes/Random_UnitCell_theta_10.npy")

    input_rr = np.load(PATH_rr)
    input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1, 0)
    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(
        input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
    input_theta = np.load(PATH_theta)
    input_theta = torch.tensor(input_theta, dtype=torch.float).permute(1, 0)

    train_rr = input_rr[:ntrain]
    test_rr = input_rr[-ntest:]
    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    sigma = train_s.std()
    mu = 0  # train_s.mean()

    train_s = (train_s - mu) / sigma
    test_s = (test_s - mu) / sigma

    return train_xy, train_s, test_xy, test_s


def get_shallow_water(filename, ntrain, ntest, min_sub=1):
    path_to_file = os.path.join(
        filename, "../shallow_water/data_t0180_freq6_N1200.h5")
    rf = h5py.File(path_to_file, "r")

    # shape (N, T, long, lat)
    # shape (1200, 6, 256, 128)

    initial_time = 0
    target_time = 5

    height_scale = 3 * 1e3
    vorticity_scale = 2

    height = torch.Tensor(rf["height"][()])
    vorticity = torch.Tensor(rf["vorticity"][()])

    # permute long and lat
    # create an extra dimension
    height = (height_scale * height).permute(0, 1, 3, 2).unsqueeze(-1)
    vorticity_scale = (vorticity_scale * vorticity).permute(0,
                                                            1, 3, 2).unsqueeze(-1)

    x_train = torch.cat(
        [height[:ntrain, initial_time], vorticity_scale[:ntrain, initial_time]], axis=-1
    )
    y_train = torch.cat(
        [height[:ntrain, target_time], vorticity_scale[:ntrain, target_time]], axis=-1
    )
    x_test = torch.cat(
        [height[-ntest:, initial_time], vorticity_scale[-ntest:, initial_time]], axis=-1
    )
    y_test = torch.cat(
        [height[-ntest:, target_time], vorticity_scale[-ntest:, target_time]], axis=-1
    )

    return x_train, y_train, x_test, y_test

def get_sst_11_22(filename, seq_inter_len=11, seq_extra_len=11):
   
    train_path = str(filename)
    h5_data = h5py.File(train_path, 'r')
    data = h5_data['mygroup']['mydataset'][:]
    h5_data.close()
    data = data.transpose(2,1,0)
    # 4459 * 64 * 64
    # 4459 = 4444 + 15 = 202 * 22 + 15 = (180 + 22) * 22 + 15
    data = data[:4444].reshape(202, 22, 64, 64) 

    u_train = torch.tensor(data[:180,:,:,:]).float()
    u_test = torch.tensor(data[180:,:,:,:]).float()

    u_min = u_train.min()
    u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())


    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 22)

def get_navier_stokes_fno(filename, ntrain=1000, ntest=200, sequence_length=None, index_start=None):
    # reader = MatReader(data_dir / "NavierStokes_V1e-5_N1200_T20.mat")
    reader = MatReader(filename)
    u = reader.read_field("u")

    # u of shape (N, Dx, Dy, T)
    u_train = u[:ntrain, :]
    u_test = u[-ntest:, :]

    print('before index start', u_train.shape)
    
    if index_start is not None:
        u_train = u_train[..., index_start:]
        u_test = u_test[..., index_start:]

    print('seq length', u_train.shape)

    if sequence_length is not None:
        # this creates sub_trajectories according of size sequence_length. Increasing the number of samples.
        u_train = einops.rearrange(
            u_train, "b w h (d t)-> (b d) w h t", t=sequence_length
        )
        u_test = einops.rearrange(
            u_test, "b w h (d t)-> (b d) w h t", t=sequence_length
        )
    print('unsqueeze', u_train.shape)
    # return u_train, u_test with shape (N, Dx, Dy, 1, Seq)
    return u_train.unsqueeze(-2), u_test.unsqueeze(-2)


def get_navier_stokes_nms_total64(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]


    data_train = np.load(train_path)
    data_test = np.load(test_path)

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    # u_train = torch.tensor(
    #     np.concatenate(
    #         list(
    #             map(
    #                 lambda key: np.array(data_train[key]["data"]),
    #                 data_train.keys(),
    #             )
    #         )
    #     )
    # )
    # u_test = torch.tensor(
    #     np.concatenate(
    #         list(
    #             map(
    #                 lambda key: np.array(data_test[key]["data"]),
    #                 data_test.keys(),
    #             )
    #         )
    #     )
    # )
    # debug
    # u_train = torch.tensor(data_train[0:1,16:,:-1,:-1,0])
    # u_min = u_train.min()
    # u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = torch.tensor(data_test[0:1,16:,:-1,:-1,0])
    # u_test = (u_test-u_min) / (u_max-u_min)
    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    u_train = (u_train-u_min) / (u_max-u_min)
    u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    # u_train = u_train[:, 24:]
    # u_test = u_test[:, 24:]

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 64)

def get_navier_stokes_nms(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]


    data_train = np.load(train_path)
    data_test = np.load(test_path)

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    # u_train = torch.tensor(
    #     np.concatenate(
    #         list(
    #             map(
    #                 lambda key: np.array(data_train[key]["data"]),
    #                 data_train.keys(),
    #             )
    #         )
    #     )
    # )
    # u_test = torch.tensor(
    #     np.concatenate(
    #         list(
    #             map(
    #                 lambda key: np.array(data_test[key]["data"]),
    #                 data_test.keys(),
    #             )
    #         )
    #     )
    # )
    # debug
    # u_train = torch.tensor(data_train[0:1,16:,:-1,:-1,0])
    # u_min = u_train.min()
    # u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = torch.tensor(data_test[0:1,16:,:-1,:-1,0])
    # u_test = (u_test-u_min) / (u_max-u_min)
    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    u_train = (u_train-u_min) / (u_max-u_min)
    u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    u_train = u_train[:, :40]
    u_test = u_test[:, :40]
    
    # lastT40
    # u_train = u_train[:, 24:]
    # u_test = u_test[:, 24:]

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_nms_last40_total60(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, 20:]
    u_test = u_test[:, 20:]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_nms_last40_total64(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    u_train = (u_train-u_min) / (u_max-u_min)
    u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, 24:]
    u_test = u_test[:, 24:]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_nms_first40_total64(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    u_train = (u_train-u_min) / (u_max-u_min)
    u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, :40]
    u_test = u_test[:, :40]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def get_navier_stokes_nms_first40_total64_wonorm(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    if not os.path.exists(train_path):
        print(f'path do not exists: {train_path}')
        train_path = train_path.replace('nvme4a', 'home1')
        print(f'switch to in xiaobai: {train_path}')
        assert os.path.exists(train_path)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, :40]
    u_test = u_test[:, :40]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_nms_last40_total64_wonorm(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    if not os.path.exists(train_path):
        print(f'path do not exists: {train_path}')
        train_path = train_path.replace('nvme4a', 'home1')
        print(f'switch to in xiaobai: {train_path}')
        assert os.path.exists(train_path)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, 24:]
    u_test = u_test[:, 24:]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_nms_last40_total70(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    test_path = train_path[:-4] + '_test' + train_path[-4:]

    data_train = np.load(train_path)
    data_test = np.load(test_path)

    u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    # u_train = (u_train-u_min) / (u_max-u_min)
    # u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    u_train = u_train[:, 30:]
    u_test = u_test[:, 30:]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_KS_fixed_viscosity_fixed_region(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/KS_train_fixed_viscosity_fixed_region.h5'
    test_path = str(filename) + '/KS_test_fixed_viscosity_fixed_region.h5'
    valid_path = str(filename) + '/KS_valid_fixed_viscosity_fixed_region.h5'

    u_train = h5py.File(train_path, 'r')['train']['pde_140-256'][:]
    u_valid = h5py.File(valid_path, 'r')['valid']['pde_640-256'][:]#[:, ::4]
    u_test = h5py.File(test_path, 'r')['test']['pde_640-256'][:]#[:, ::4]
    
    print(u_train.shape, u_valid.shape, u_test.shape)
    
    u_train = torch.tensor(u_train, dtype=torch.float32).permute(0, 2, 1).unsqueeze(-2)
    u_valid = torch.tensor(u_valid, dtype=torch.float32).permute(0, 2, 1).unsqueeze(-2)
    u_test = torch.tensor(u_test, dtype=torch.float32).permute(0, 2, 1).unsqueeze(-2)

    print(u_train.shape, u_valid.shape, u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    # return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)
    return u_train, u_valid, u_test
    # return u_train, u_test, u_valid

def get_navier_stokes_dino(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_stokes_dino_v0/navier_1e-3_256_train.shelve'
    test_path = str(filename) + '/navier_stokes_dino_v0/navier_1e-3_256_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def get_navier_stokes_dino_first40_T100(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_256_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    u_train = u_train[..., :40]
    u_test = u_test[..., :40]

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_dino_mid40_T100(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_256_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    u_train = u_train[..., 30:70]
    u_test = u_test[..., 30:70]
    
    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def get_navier_stokes_dino_T100(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_256_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # u_train = u_train[..., 30:70]
    # u_test = u_test[..., 30:70]
    
    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 100)



def get_navier_stokes_dino_first40_T100_2048(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_2048_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"][:, :40, ::4, ::4]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"][:, :40, ::4, ::4]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # u_train = u_train[..., :40]
    # u_test = u_test[..., :40]

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def get_navier_stokes_dino_mid40_T100_2048(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_2048_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"][:, 30:70, ::4, ::4]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"][:, 30:70, ::4, ::4]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # u_train = u_train[..., 30:70]
    # u_test = u_test[..., 30:70]
    
    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def get_navier_stokes_dino_T100_2048(filename, seq_inter_len=20, seq_extra_len=20):
    # train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    # test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    train_path = str(filename) + '/navier_1e-3_2048_T100_train.shelve'
    test_path = str(filename) + '/navier_1e-3_256_T100_test.shelve'

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_min = u_train.min()
    # u_max = u_train.max()
    # print(u_min, u_max, u_test.min(), u_test.max())

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # u_train = u_train[..., 30:70]
    # u_test = u_test[..., 30:70]
    
    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 100)

def get_navier_stokes_dino_last40_total64(filename, seq_inter_len=20, seq_extra_len=20):
   
    train_path = str(filename)
    # test_path = train_path[:-4] + '_test' + train_path[-4:]
    test_path = train_path.replace('train', 'test')

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train = np.load(train_path)
    # data_test = np.load(test_path)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    # u_train = torch.tensor(data_train[:,:,:-1,:-1,0])
    # u_test = torch.tensor(data_test[:,:,:-1,:-1,0])

    u_min = u_train.min()
    u_max = u_train.max()
    u_train = (u_train-u_min) / (u_max-u_min)
    u_test = (u_test-u_min) / (u_max-u_min)
    print(u_min, u_max, u_test.min(), u_test.max())

    # firstT40
    # u_train = u_train[:, :40]
    # u_test = u_test[:, :40]
    
    # lastT40
    # u_train = u_train[:, 24:]
    # u_test = u_test[:, 24:]

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    print(u_train.shape)
    print(u_test.shape)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)

def split_data(u_train, u_test, seq_inter_len, seq_extra_len, total_seq):
    
    if (seq_inter_len is not None) & (seq_extra_len is not None):
        if total_seq % (seq_inter_len+seq_extra_len) == 0:
            u_train = einops.rearrange(u_train, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
            u_test = einops.rearrange(u_test, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
        elif total_seq % (seq_inter_len+seq_extra_len) != 0:
            u_train = u_train[..., total_seq - total_seq // (seq_inter_len+seq_extra_len) * (seq_inter_len+seq_extra_len):]
            u_test = u_test[..., total_seq - total_seq // (seq_inter_len+seq_extra_len) * (seq_inter_len+seq_extra_len):]
            u_train = einops.rearrange(u_train, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
            u_test = einops.rearrange(u_test, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
    u_eval_extrapolation = u_train
    u_train = u_train[..., :seq_inter_len]
    return u_train, u_eval_extrapolation, u_test


def get_shallow_water_dino(filename, seq_inter_len = 20, seq_extra_len = 20):
    train_path = str(filename) + "/shallow_water_16_160_128_256_train.h5"
    test_path = str(filename) + "/shallow_water_2_160_128_256_test.h5"

    with h5py.File(train_path, "r") as f:
        vorticity_train = f["vorticity"][()]
        height_train = f["height"][()]

    with h5py.File(test_path, "r") as f:
        vorticity_test = f["vorticity"][()]
        height_test = f["height"][()]

    # train_path = os.path.join(str(filename),"shallow_water_train")
    # test_path = os.path.join(str(filename),"shallow_water_test")

    # vorticity_train = []
    # height_train = []
    # for traj_id in range(8):
    #     with h5py.File(os.path.join(train_path, f'traj_{traj_id:04d}.h5'), "r") as f:
    #         vorticity_train.append(f["tasks/vorticity"][()])
    #         height_train.append(f["tasks/height"][()])
    # vorticity_train = np.stack(vorticity_train)
    # height_train = np.stack(height_train)

    # vorticity_test = []
    # height_test = []
    # for traj_id in range(2):
    #     with h5py.File(os.path.join(test_path, f'traj_{traj_id:04d}.h5'), "r") as f:
    #         vorticity_test.append(f["tasks/vorticity"][()])
    #         height_test.append(f["tasks/height"][()])
    # vorticity_test = np.stack(vorticity_test)
    # height_test = np.stack(height_test)

    print(f'train {vorticity_train.shape}, {height_train.shape}')
    print(f'test {vorticity_test.shape}, {height_test.shape}')
    # shape (N, T, long, lat)
    # train shape (16, 160, 256, 128)
    # test shape (2, 160, 256, 128)

    height_scale = 3 * 1e3
    vorticity_scale = 2

    vorticity_train = torch.from_numpy(
        vorticity_train).float() * vorticity_scale
    vorticity_test = torch.from_numpy(vorticity_test).float() * vorticity_scale

    height_train = torch.from_numpy(height_train).float() * height_scale
    height_test = torch.from_numpy(height_test).float() * height_scale

    u_train = torch.cat([height_train.unsqueeze(-1),
                        vorticity_train.unsqueeze(-1)], axis=-1)
    u_test = torch.cat(
        [height_test.unsqueeze(-1), vorticity_test.unsqueeze(-1)], axis=-1)

    u_train = einops.rearrange(u_train, 'b t long lat c -> b lat long c t')
    u_test = einops.rearrange(u_test, 'b t long lat c -> b lat long c t')
    
    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 160)

class MatReader(object):
    """Loader for navier-stokes data"""

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except BaseException:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x


def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def operator_subsample(x, sub=1, missing_batch=False):
    """
    WARNING: This functions does not work for graph data.
    Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled, of shape (N, Dx, Dy, C)
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)

    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 3:
            x = x[:, ::sub]
        if x.ndim == 4:
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]


    if missing_batch:
        x = x.squeeze(0)
    return x


def operator_different_subsample(x, y, grid, draw_ratio):
    """
    WARNING: This functions does not work for graph data.
    Performs different subsampling per sample for operator data.
    Args:
        data (torch.Tensor): univariate time series (batch_size, num_points, num_channels)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    N = x.shape[0]
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])
    grid = grid.reshape(grid.shape[0], -1, grid.shape[-1])

    in_channels = x.shape[-1]
    out_channels = y.shape[-1]
    input_dim = grid.shape[-1]
    partial_grid_size = int(draw_ratio * grid.shape[1])

    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)

    small_x = torch.gather(
        x, 1, permutations.unsqueeze(-1).repeat(1, 1, in_channels))
    small_y = torch.gather(
        y, 1, permutations.unsqueeze(-1).repeat(1, 1, out_channels))
    small_grid = torch.gather(
        grid, 1, permutations.unsqueeze(-1).repeat(1, 1, input_dim)
    )

    return small_x, small_y, small_grid, permutations


def dynamics_subsample(x, sub=1, missing_batch=False):
    """
    WARNING: This functions does not work for graph data.

    Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled, of shape (N, Dx, Dy, C, T)
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)
    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 4:  # 1D data (N, Dx, C, T)
            x = x[:, ::sub]
        if x.ndim == 5:  # 2D data (N, Dx, Dy, C, T)
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = einops.rearrange(
            x, "b ... c t -> b (...) c t"
        )  # x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]

    if missing_batch:
        x = x.squeeze(0)
    return x

def dynamics_different_subsample(u, grid, draw_ratio):
    """
    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels, T)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    u = einops.rearrange(u, "b ... c t -> b (...) c t")
    grid = einops.rearrange(grid, "b ... c t -> b (...) c t")

    N = u.shape[0]
    C = u.shape[-2]
    dims = grid.shape[-2]
    T = u.shape[-1]
    input_dim = grid.shape[-2]
    partial_grid_size = int(draw_ratio * grid.shape[1])
    
    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)
    small_u = torch.gather(u, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, C, T))
    small_grid = torch.gather(grid, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, dims, T))

    return small_u, small_grid, permutations

def dynamics_different_subsample_two_grid(u, grid, draw_ratio, u1=None, g1=None, u2=None, g2=None):
    """
    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels, T)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    u = einops.rearrange(u, "b ... c t -> b (...) c t")
    grid = einops.rearrange(grid, "b ... c t -> b (...) c t")

    N = u.shape[0]
    C = u.shape[-2]
    dims = grid.shape[-2]
    T = u.shape[-1]
    input_dim = grid.shape[-2]
    partial_grid_size = int(draw_ratio * grid.shape[1])
    
    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)
    small_u = torch.gather(u, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, C, T))
    small_grid = torch.gather(grid, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, dims, T))

    small_u1 = None 
    small_g1 = None 
    small_u2 = None 
    small_g2 = None 

    if u1 is not None:
        u1 = einops.rearrange(u1, "b ... c t -> b (...) c t")
        g1 = einops.rearrange(g1, "b ... c t -> b (...) c t")
        T1 = u1.shape[-1]
        N1 = u1.shape[0]
        small_u1 = torch.gather(u1, 1, permutations[0:1].unsqueeze(-1).unsqueeze(-1).repeat( N1, 1, C, T1))
        small_g1 = torch.gather(g1, 1, permutations[0:1].unsqueeze(-1).unsqueeze(-1).repeat( N1, 1, dims, T1))
    if u2 is not None:
        u2 = einops.rearrange(u2, "b ... c t -> b (...) c t")
        g2 = einops.rearrange(g2, "b ... c t -> b (...) c t")
        T2 = u2.shape[-1]
        N2 = u2.shape[0]
        small_u2 = torch.gather(u2, 1, permutations[0:1].unsqueeze(-1).unsqueeze(-1).repeat( N2, 1, C, T2))
        small_g2 = torch.gather(g2, 1, permutations[0:1].unsqueeze(-1).unsqueeze(-1).repeat( N2, 1, dims, T2))

    return small_u, small_grid, small_u1, small_g1, small_u2, small_g2, permutations

def shape2coordinates(spatial_shape, max_value=1.0):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        # coords.append(torch.linspace(0.0, max_value, spatial_shape[i]+1)[:-1])
        coords.append(torch.linspace(0.0, max_value, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def shape2circular_coordinates(spatial_shape):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 2*np.pi, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)

    new_coords = torch.zeros(*coords.shape[:-1], 2)
    new_coords[..., 0] = torch.sin(coords[..., 0])
    new_coords[..., 1] = torch.cos(coords[..., 0])
    return new_coords

def shape2spherical_coordinates(spatial_shape):
    """Returns spherical coordinates on a uniform latitude and longitude grid.
    Args:
        spatial_shape (tuple of int): Tuple (num_lats, num_lons) containing
            number of latitudes and longitudes in grid.
    """
    num_lats, num_lons = spatial_shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = torch.linspace(90.0, -90.0, num_lats)
    longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
    # Create a grid of latitude and longitude values (num_lats, num_lons)
    longitude_grid, latitude_grid = torch.meshgrid(
        longitude, latitude, indexing="xy")
    # Create coordinate tensor
    # Spherical coordinates have 3 dimensions
    coordinates = torch.zeros(latitude_grid.shape + (3,))
    long_rad = deg_to_rad(longitude_grid)
    lat_rad = deg_to_rad(latitude_grid)
    coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
    coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
    coordinates[..., 2] = torch.sin(lat_rad)
    
    long_rad = (long_rad-long_rad.min()) / (long_rad.max()-long_rad.min())
    lat_rad = (lat_rad-lat_rad.min()) / (lat_rad.max()-lat_rad.min())
    coordinates = torch.cat([coordinates, lat_rad.unsqueeze(-1), long_rad.unsqueeze(-1)], dim=-1)
    return coordinates


def deg_to_rad(degrees):
    return torch.pi * degrees / 180.0


def rad_to_deg(radians):
    return 180.0 * radians / torch.pi


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates
