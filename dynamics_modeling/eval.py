import torch
import torch.nn as nn
import einops
from torchdiffeq import odeint
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from visualize import write_image_pair 

def batch_eval_loop(model, inr, loader, timestamps, detailed_mse, 
                     n, multichannel, z_mean, z_std, dataset_name, 
                     interpolation_seq, n_cond, visual_first=0, visual_path=''):
    interpolation_seq = interpolation_seq - n_cond
    visual=True
    if interpolation_seq:
        pred_mse_inter = 0
        code_mse_inter = 0
        pred_mse_extra = 0
        code_mse_extra = 0
        total_pred_mse = 0  # Rename to avoid confusion with pred_mse in the loop
        
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

            code_mse_inter += loss_inter.item() * n_samples
            code_mse_extra += loss_extra.item() * n_samples
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
                gt_reshape = gt.permute(0,4,1,2,3).detach().cpu().numpy()
                pred_reshape = pred.permute(0,4,1,2,3).detach().cpu().numpy()
                images_reshape = images.permute(0,4,1,2,3).detach().cpu().numpy()
                for visual_idx in range(visual_first):
                    # print(pred_reshape.shape, images_reshape.shape)
                    write_image_pair(images_reshape[visual_idx], pred_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}.png', cmap='twilight_shifted', divider=2)
                    write_image_pair(images_reshape[visual_idx], gt_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_gt.png', cmap='twilight_shifted', divider=2)
                visual=False 
        pred_mse_inter /= n
        pred_mse_extra /= n
        code_mse_inter /= n
        code_mse_extra /= n      
        total_pred_mse /= n
        return pred_mse_inter, code_mse_inter, pred_mse_extra, code_mse_extra, total_pred_mse, detailed_mse
    
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

        code_mse /= n
        total_pred_mse /= n
        return total_pred_mse, code_mse, detailed_mse
