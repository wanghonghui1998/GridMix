import torch
import torch.nn as nn
import numpy as np 
import einops
from torchdiffeq import odeint
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from visualize import write_image_pair, write_image  

def batch_eval_loop(model, inr, loader, timestamps, detailed_mse, 
                     n, multichannel, z_mean, z_std, dataset_name, 
                     interpolation_seq, n_cond, code_dim, visual_first=0, visual_path='', visual_mod=0):
    interpolation_seq = interpolation_seq - n_cond
    visual=True
    # print(code_dim)
    if interpolation_seq:
        pred_mse_inter = 0
        code_mse_inter = 0
        pred_mse_extra = 0
        code_mse_extra = 0
        total_pred_mse = 0  # Rename to avoid confusion with pred_mse in the loop
      
        code_mse_inter_code = 0
        code_mse_extra_code = 0
        code_mse_inter_grid = 0
        code_mse_extra_grid = 0
      
        for images, modulations, coords, idx in loader:
            model.eval()
            images = images.cuda()
            modulations = modulations.cuda()
            coords = coords.cuda()
            n_samples = images.shape[0]

            if multichannel:
                modulations = einops.rearrange(modulations, "b l c t -> b (l c) t")

            with torch.no_grad():
                modulations = modulations[...,n_cond:]
                images = images[...,n_cond:]
                coords = coords[...,n_cond:]

                z_pred = ode_scheduling(
                    odeint, model, modulations, timestamps[n_cond:], epsilon=0
                )
            loss = ((z_pred - modulations) ** 2)
            loss_inter = loss[..., :interpolation_seq].mean()
            loss_extra = loss[..., interpolation_seq:].mean()
            loss_inter_code = loss[..., :code_dim, :interpolation_seq].mean()
            loss_extra_code = loss[..., :code_dim, interpolation_seq:].mean()
            loss_inter_grid = loss[..., code_dim:, :interpolation_seq].mean()
            loss_extra_grid = loss[..., code_dim:, interpolation_seq:].mean()

            code_mse_inter += loss_inter.item() * n_samples
            code_mse_extra += loss_extra.item() * n_samples
            code_mse_inter_code += loss_inter_code.item() * n_samples
            code_mse_extra_code += loss_extra_code.item() * n_samples
            code_mse_inter_grid += loss_inter_grid.item() * n_samples
            code_mse_extra_grid += loss_extra_grid.item() * n_samples
            with torch.no_grad():
                pred = get_reconstructions(
                    inr, coords, z_pred, z_mean, z_std, dataset_name
                )
            pred_mse = ((pred - images) ** 2)
            pred_mse_inter_local = pred_mse[..., :interpolation_seq].mean()
            pred_mse_extra_local = pred_mse[..., interpolation_seq:].mean()
            total_pred_mse += pred_mse.mean() * n_samples
            pred_mse_inter += pred_mse_inter_local.item() * n_samples
            pred_mse_extra += pred_mse_extra_local.item() * n_samples

            if multichannel:
                detailed_mse.aggregate(pred, images)
            
            if visual and visual_first > 0:
                gt = get_reconstructions(
                    inr, coords, modulations, z_mean, z_std, dataset_name
                )
                if gt.dim() == 5:
                    gt_reshape = gt.permute(0,4,1,2,3).detach().cpu().numpy()
                    pred_reshape = pred.permute(0,4,1,2,3).detach().cpu().numpy()
                    images_reshape = images.permute(0,4,1,2,3).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        # print(pred_reshape.shape, images_reshape.shape)
                        write_image_pair(images_reshape[visual_idx], pred_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}.png', cmap='twilight_shifted', divider=2)
                        error = np.abs(images_reshape[visual_idx] - pred_reshape[visual_idx])
                        write_image(error, error, 0, path=visual_path+f'error_{visual_idx}.png', cmap='twilight_shifted', divider=2)
                        write_image_pair(images_reshape[visual_idx], gt_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_gt.png', cmap='twilight_shifted', divider=2)

                if visual_mod > 0:
                    modulations_v = modulations.permute(0,2,1)
                    modulations_v = modulations_v[...,code_dim:].reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
                    divider = 2 * modulations_v.shape[2]
                    modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        write_image(modulations_v[visual_idx], modulations_v[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)

                    modulations_v = z_pred.permute(0,2,1)
                    modulations_v = modulations_v[...,code_dim:].reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
                    divider = 2 * modulations_v.shape[2]
                    modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        write_image(modulations_v[visual_idx], modulations_v[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod_pred.png', cmap='twilight_shifted', divider=divider)

                visual=False 
        pred_mse_inter /= n
        pred_mse_extra /= n
        code_mse_inter /= n
        code_mse_extra /= n      
        total_pred_mse /= n

        code_mse_inter_code /= n
        code_mse_extra_code /= n
        code_mse_inter_grid /= n
        code_mse_extra_grid /= n
        return pred_mse_inter, code_mse_inter, pred_mse_extra, code_mse_extra, total_pred_mse, detailed_mse, code_mse_inter_code, code_mse_extra_code, code_mse_inter_grid, code_mse_extra_grid
    
    else:
        total_pred_mse = 0  # Use a different variable name for clarity
        code_mse = 0
        for images, modulations, coords, idx in loader:
            model.eval()
            images = images.cuda()
            modulations = modulations.cuda()
            coords = coords.cuda()
            n_samples = images.shape[0]

            if multichannel:
                modulations = einops.rearrange(modulations, "b l c t -> b (l c) t")

            with torch.no_grad():
                z_pred = ode_scheduling(
                    odeint, model, modulations, timestamps, epsilon=0
                )
            loss = ((z_pred - modulations) ** 2).mean()
            code_mse += loss.item() * n_samples

            pred = get_reconstructions(
                inr, coords, z_pred, z_mean, z_std, dataset_name
            )
            total_pred_mse += ((pred - images) ** 2).mean().item() * n_samples

            if multichannel:
                detailed_mse.aggregate(pred, images)

            if visual and visual_first > 0:
                gt = get_reconstructions(
                    inr, coords, modulations, z_mean, z_std, dataset_name
                )
                if gt.dim() == 5:
                    gt_reshape = gt.permute(0,4,1,2,3).detach().cpu().numpy()
                    pred_reshape = pred.permute(0,4,1,2,3).detach().cpu().numpy()
                    images_reshape = images.permute(0,4,1,2,3).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        # print(pred_reshape.shape, images_reshape.shape)
                        write_image_pair(images_reshape[visual_idx], pred_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}.png', cmap='twilight_shifted', divider=1)
                        write_image_pair(images_reshape[visual_idx], gt_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_gt.png', cmap='twilight_shifted', divider=1)
                if visual_mod > 0:
                    modulations_v = modulations.permute(0,2,1)
                    modulations_v = modulations_v.reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
                    divider = 1 * modulations_v.shape[2]
                    modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        write_image(modulations_v[visual_idx], modulations_v[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)


                visual=False 

        code_mse /= n
        total_pred_mse /= n
        return total_pred_mse, code_mse, detailed_mse
