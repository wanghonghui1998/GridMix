from collections import OrderedDict
from functools import partial

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP

import coral.losses as losses

# adapted from https://github.com/EmilienDupont/coinpp/blob/main/coinpp/metalearning.py

def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
                loss_type,
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                inner_lr,
                is_train,
                gradient_checkpointing,
                loss_type,
            )
    return fitted_modulations


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)
    if loss_type == "mse":
        element_loss_fn = losses.per_element_mse_fn
    elif loss_type == "bce":
        element_loss_fn = losses.per_element_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        element_loss_fn = partial(
            losses.per_element_multi_scale_fn,
            loss_name=loss_name,
            last_element=False,
        )

    N, C = features.shape[0], features.shape[-1]

    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        features_recon = func_rep.modulated_forward(coordinates, modulations)

        loss = element_loss_fn(features_recon, features).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    return modulations - inner_lr * grad

def inner_loop_grad_mask_sep_lr(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    inner_lr_grid,
    gradient_mask,
    code_dim=128,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step_grad_mask_sep_lr,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
                loss_type,
            )
        else:
            fitted_modulations = inner_loop_step_grad_mask_sep_lr(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                inner_lr,
                inner_lr_grid,
                gradient_mask,
                code_dim,
                is_train,
                gradient_checkpointing,
                loss_type,
            )
    return fitted_modulations

def inner_loop_step_grad_mask_sep_lr(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_lr,
    inner_lr_grid,
    gradient_mask,
    code_dim=128,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)
    if loss_type == "mse":
        element_loss_fn = losses.per_element_mse_fn
    elif loss_type == "bce":
        element_loss_fn = losses.per_element_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        element_loss_fn = partial(
            losses.per_element_multi_scale_fn,
            loss_name=loss_name,
            last_element=False,
        )

    N, C = features.shape[0], features.shape[-1]

    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        features_recon = func_rep.modulated_forward(coordinates, modulations)

        loss = element_loss_fn(features_recon, features).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    update = torch.cat([inner_lr * grad[...,:code_dim], inner_lr_grid * grad[...,code_dim:]], dim=-1)

    return modulations - gradient_mask * update

def outer_step_dino(
    func_rep,
    coordinates,
    features,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    # func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    # modulations = modulations.requires_grad_()

    # feat = features.clone()
    # coords = coordinates.clone()

    # # Run inner loop
    # modulations = inner_loop(
    #     func_rep,
    #     modulations,
    #     coords,
    #     feat,
    #     inner_steps,
    #     inner_lr,
    #     is_train,
    #     gradient_checkpointing,
    #     loss_type,
    # )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        # "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs


def outer_step_metagrid_twoview_rand(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
    code_dim=128,
    grid_channel=3,
    grid_size=32,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.cat([modulations, modulations], dim=0)
    modulations = modulations.requires_grad_()
    num_points = features.shape[1]

    perm = torch.randperm(num_points)
    feat_view1_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_view1_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    feat_view1_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    coords_view1_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()
    perm = torch.randperm(num_points)
    feat_view2_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_view2_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    feat_view2_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    coords_view2_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    feat_train = torch.cat([feat_view1_train, feat_view2_train], dim=0)
    feat_val = torch.cat([feat_view1_val, feat_view2_val], dim=0)
    coords_train = torch.cat([coords_view1_train, coords_view2_train], dim=0)
    coords_val = torch.cat([coords_view1_val, coords_view2_val], dim=0)
    
    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords_train, #torch.cat([coords_view1, coords_view2],dim=0),
        feat_train, #torch.cat([feat_view1, feat_view2],dim=0),
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()
        modulations_con = modulations[...,code_dim:].reshape(-1, grid_channel, grid_size*grid_size)
        mod1, mod2 = torch.chunk(modulations_con, 2, dim=0)
        # loss_mod_con = 2 * torch.linalg.norm(mod1-mod2, 2, dim=-1) / (torch.linalg.norm(mod1, 2, dim=-1) + torch.linalg.norm(mod2, 2, dim=-1) + 1e-10)
        loss_mod_con1 = torch.linalg.norm(mod1-mod2.detach(), 2, dim=-1) / (torch.linalg.norm(mod2.detach(), 2, dim=-1) + 1e-10)
        loss_mod_con2 = torch.linalg.norm(mod2-mod1.detach(), 2, dim=-1) / (torch.linalg.norm(mod1.detach(), 2, dim=-1) + 1e-10)
        loss_mod_con = (loss_mod_con1+loss_mod_con2).mean()
    
    outputs = {
        "loss": loss,
        "loss_mod_con": loss_mod_con,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid_twoview(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.cat([modulations, modulations], dim=0)
    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_view1 = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_view1 = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()

    feat_view2 = features[:, perm[int(grid_ratio*num_points):]].clone()
    coords_view2 = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        torch.cat([coords_view1, coords_view2],dim=0),
        torch.cat([feat_view1, feat_view2],dim=0),
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(torch.cat([coords_view2, coords_view1],dim=0), modulations)
        per_example_loss = loss_fn(features_recon, torch.cat([feat_view2, feat_view1],dim=0))  # features
        loss = per_example_loss.mean()
        mod1, mod2 = torch.chunk(modulations, 2, dim=0)
        loss_mod_con = torch.mean((mod1-mod2)**2)

    outputs = {
        "loss": loss,
        "loss_mod_con": loss_mod_con,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], torch.cat([feat_view2, feat_view1],dim=0)).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, torch.cat([feat_view2, feat_view1],dim=0)).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid_same_coords_teacher_boosting(
    func_rep,
    coordinates,
    features,
    func_rep_teacher,
    extra_coordinates,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
    extra_only=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    
    feat_val = feat_train.clone()
    coords_val = coords_train.clone()
    # feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    # coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep_teacher,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        False,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(False):
        extra_features_recon = func_rep_teacher.modulated_forward(extra_coordinates, modulations)
        # per_example_loss = loss_fn(features_recon, feat_val)  # features
        # loss = per_example_loss.mean()
    
    if not extra_only:
        extra_coordinates = torch.cat([coords_train, extra_coordinates], dim=1)
        extra_features_recon = torch.cat([feat_train, extra_features_recon.detach()], dim=1)
    
    modulations = torch.zeros_like(modulations).requires_grad_()
    modulations = inner_loop(
        func_rep,
        modulations,
        extra_coordinates,
        extra_features_recon,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid_same_coords_sep_lr(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    inner_lr_grid,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
    code_dim=128,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    
    feat_val = feat_train.clone()
    coords_val = coords_train.clone()
    # feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    # coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()
    gradient_mask = 1
    # Run inner loop
    modulations = inner_loop_grad_mask_sep_lr(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        inner_lr_grid,
        gradient_mask,
        code_dim,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid_same_coords_sep_lr_two_stage(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    inner_lr_grid,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
    code_dim=128,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    
    feat_val = feat_train.clone()
    coords_val = coords_train.clone()
    # feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    # coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()
    gradient_mask = torch.ones(modulations.shape[-1], device=modulations.device)
    gradient_mask[code_dim:] = 0
    # Run inner loop
    modulations = inner_loop_grad_mask_sep_lr(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        inner_lr_grid,
        gradient_mask,
        code_dim,
        is_train,
        gradient_checkpointing,
        loss_type,
    )
    gradient_mask = 1 - gradient_mask
    modulations = inner_loop_grad_mask_sep_lr(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        inner_lr_grid,
        gradient_mask,
        code_dim,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid_same_coords(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    
    feat_val = feat_train.clone()
    coords_val = coords_train.clone()
    # feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    # coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs


def outer_step_metagrid_part_coords(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()
    
    feat_val = features.clone()
    coords_val = coordinates.clone()
    # feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    # coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_metagrid(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    grid_ratio,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()
    num_points = features.shape[1]
    perm = torch.randperm(num_points)
    feat_train = features[:, perm[:int(grid_ratio*num_points)]].clone()
    coords_train = coordinates[:, perm[:int(grid_ratio*num_points)]].clone()

    feat_val = features[:, perm[int(grid_ratio*num_points):]].clone()
    coords_val = coordinates[:, perm[int(grid_ratio*num_points):]].clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords_train,
        feat_train,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords_val, modulations)
        per_example_loss = loss_fn(features_recon, feat_val)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], feat_val).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat_val).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_multiple_init(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    num_init = func_rep.latent_init.shape[0]
    batch_size = len(coordinates)
    # modulations = modulations.unsqueeze(0).repeat(num_init, 1, 1).reshape(num_init*batch_size, -1)
    modulations = modulations.unsqueeze(0).repeat(num_init, 1, 1) # einops.repeat(modulations, '... -> b ...', b=num_init)
    modulations = modulations.requires_grad_()
    modulations = modulations + func_rep.latent_init.unsqueeze(1)
    modulations = modulations.reshape(num_init*batch_size, -1)

    features = einops.repeat(features, '... -> b ...', b=num_init)
    features = einops.rearrange(features,"n b ... -> (n b) ...")

    coordinates = einops.repeat(coordinates, '... -> b ...', b=num_init)
    coordinates = einops.rearrange(coordinates,"n b ... -> (n b) ...")
  
    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        per_example_loss_min,_ = torch.min(per_example_loss.reshape(num_init, batch_size), dim=0)
        loss = per_example_loss_min.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()

    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_boosting(
    func_rep,
    extra_coordinates,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()

    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        extra_features_recon = func_rep.modulated_forward(extra_coordinates, modulations)

    coords = torch.cat([coordinates, extra_coordinates], dim=1)
    feat = torch.cat([features, extra_features_recon.detach()], dim=1)

    modulations = torch.zeros_like(modulations).requires_grad_()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_test_on_diff_grid(
    func_rep,
    coordinates,
    features,
    diff_coordinates,
    diff_features,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()

    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()
        diff_features_recon = func_rep.modulated_forward(diff_coordinates, modulations)
        diff_per_example_loss = loss_fn(diff_features_recon, diff_features)  # features
        diff_loss = diff_per_example_loss.mean()

    outputs = {
        "loss": loss,
        "diff_loss": diff_loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_kd(
    func_rep,
    teacher_func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    teacher_inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    teacher_modulations=0,
    use_rel_loss=False,
    kd_num_coords=0,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    teacher_modulations = teacher_modulations.requires_grad_()
    teacher_feat = features.clone()
    teacher_coords = coordinates.clone()
    teacher_modulations = inner_loop(
        teacher_func_rep,
        teacher_modulations,
        teacher_coords,
        teacher_feat,
        inner_steps,
        teacher_inner_lr,
        False,
        gradient_checkpointing,
        loss_type,
    )

    # coords = torch.rand_like(coordinates[:, 0:1]).repeat(kd_num_coords, 1, 1)
    coords = torch.rand((1, kd_num_coords, coordinates.shape[-1]), device=coordinates.device).repeat(coordinates.shape[0], 1, 1)
    coords = torch.cat([coordinates, coords], dim=1)
    with torch.set_grad_enabled(False):
        feat = teacher_func_rep.modulated_forward(coords, teacher_modulations)
        teacher_per_example_loss = loss_fn(feat[:,:features.shape[1]], features)  # features
        teacher_loss = teacher_per_example_loss.mean()
    
    modulations = modulations.requires_grad_()

    feat = torch.cat([features, feat[:,features.shape[1]:]], dim=1)
    # coords = torch.cat([coordinates, coords], dim=1)
    # coords = coordinates.clone()
    # feat = features.clone()
    # coords = coordinates.clone()
    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations)
        per_example_loss = loss_fn(features_recon, feat)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "teacher_loss": teacher_loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, feat).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def outer_step_extra_grid(
    func_rep,
    coordinates,
    features,
    coordinates_extra_grid,
    features_extra_grid,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    loss_type="mse",
    modulations=0,
    use_rel_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()

    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()

    with torch.set_grad_enabled(False):
        features_recon_extra_grid = func_rep.modulated_forward(coordinates_extra_grid, modulations)
        per_example_loss_extra_grid = loss_fn(features_recon_extra_grid, features_extra_grid)  # features
        loss_extra_grid = per_example_loss_extra_grid.mean()


    outputs = {
        "loss": loss,
        "loss_extra_grid": loss_extra_grid,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs

def graph_inner_loop(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                graph_inner_loop_step,
                func_rep,
                fitted_modulations,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
                loss_type,
            )
        else:
            fitted_modulations = graph_inner_loop_step(
                func_rep,
                fitted_modulations,
                coords,
                features,
                batch_index,
                inner_lr,
                is_train,
                gradient_checkpointing,
                loss_type,
            )
    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    last_element=False,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = modulations.shape[0]
    if loss_type == "mse":
        element_loss_fn = losses.per_element_mse_fn
    elif loss_type == "nll":
        element_loss_fn = losses.per_element_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        element_loss_fn = partial(
            losses.per_element_multi_scale_fn,
            loss_name=loss_name,
            last_element=last_element,
        )

    loss = 0
    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch

        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        loss = ((features_recon - features) ** 2).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    return modulations - inner_lr * grad

def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    use_rel_loss=False,
    loss_type="mse",
    detach_modulations=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)

    func_rep.zero_grad()
    batch_size = len(graph)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.zeros_like(graph.modulations).requires_grad_()
    coords = graph.pos
    features = graph.images

    # Run inner loop
    modulations = graph_inner_loop(
        func_rep,
        modulations,
        coords,
        features,
        graph.batch,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    if detach_modulations:
        modulations = modulations.detach()  # 1er ordre

    loss = 0
    batch_size = modulations.shape[0]

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[graph.batch])
        loss = ((features_recon - features) ** 2).mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = (
            features_recon[-1] if "multiscale" in loss_type else features_recon
        )

    if use_rel_loss:
        rel_loss = (
            losses.batch_mse_rel_fn(features_recon[-1], features).mean()
            if "multiscale" in loss_type
            else losses.batch_mse_rel_fn(features_recon, features).mean()
        )
        outputs["rel_loss"] = rel_loss

    return outputs
