import torch
import torch.nn as nn
import numpy as np 
import einops
from torchdiffeq import odeint
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from visualize import write_image_pair, write_image, write_error, write_prediction, write_prediction_sw

def batch_eval_loop(model, inr, loader, timestamps, detailed_mse, 
                     n, multichannel, z_mean, z_std, dataset_name, 
                     interpolation_seq, n_cond, code_dim=0, visual_first=0, visual_path='', visual_mod=0):
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
                origin = get_reconstructions(
                    inr, coords, torch.zeros_like(modulations), torch.zeros_like(z_mean), torch.ones_like(z_std), dataset_name
                )
                if gt.dim() == 5:
                    gt_reshape = gt.permute(0,4,1,2,3).detach().cpu().numpy()
                    origin_reshape = origin.permute(0,4,1,2,3).detach().cpu().numpy()
                    pred_reshape = pred.permute(0,4,1,2,3).detach().cpu().numpy()
                    images_reshape = images.permute(0,4,1,2,3).detach().cpu().numpy()
                    for visual_idx in range(visual_first):
                        # print(pred_reshape.shape, images_reshape.shape)
                        # write_image_pair(images_reshape[visual_idx], pred_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}.png', cmap='twilight_shifted', divider=2)
                        # error = np.abs(images_reshape[visual_idx] - pred_reshape[visual_idx])
                        # write_image(error, error, 0, path=visual_path+f'error_{visual_idx}.png', cmap='twilight_shifted', divider=2)
                        # write_image_pair(images_reshape[visual_idx], gt_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_gtmod.png', cmap='twilight_shifted', divider=2)
                        # write_image_pair(images_reshape[visual_idx], origin_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_origin.png', cmap='twilight_shifted', divider=2)
                        write_prediction(images_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 0, path=visual_path+f'gt_{visual_idx}.pdf', cmap='twilight_shifted', divider=1)
                        write_prediction(pred_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 0, path=visual_path+f'pred_{visual_idx}.pdf', cmap='twilight_shifted', divider=1)
                        error = np.abs(images_reshape[visual_idx,4::5] - pred_reshape[visual_idx,4::5])
                        write_error(error, error, 0, path=visual_path+f'error_{visual_idx}.png', cmap='twilight_shifted', divider=1)
                        # write_image_pair(images_reshape[visual_idx], gt_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_gtmod.png', cmap='twilight_shifted', divider=2)
                        # write_image_pair(images_reshape[visual_idx], origin_reshape[visual_idx], 0, path=visual_path+f'_{visual_idx}_origin.png', cmap='twilight_shifted', divider=2)


                if visual_mod > 0:
                    modulations_v = modulations.permute(0,2,1)
                    modulations_v = modulations_v[...,code_dim:].reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
                    divider = 2 * modulations_v.shape[2]
                    modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                    modulations_v_pred = z_pred.permute(0,2,1)
                    modulations_v_pred = modulations_v_pred[...,code_dim:].reshape(modulations_v_pred.shape[0], modulations_v_pred.shape[1], -1, visual_mod, visual_mod)
                    modulations_v_pred = modulations_v_pred.permute(0,2,1,3,4).reshape(modulations_v_pred.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                     
                    
                    for visual_idx in range(visual_first):
                        write_image_pair(modulations_v[visual_idx], modulations_v_pred[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)


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
                    modulations_v_pred = z_pred.permute(0,2,1)
                    modulations_v_pred = modulations_v_pred.reshape(modulations_v_pred.shape[0], modulations_v_pred.shape[1], -1, visual_mod, visual_mod)
                    modulations_v_pred = modulations_v_pred.permute(0,2,1,3,4).reshape(modulations_v_pred.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                     
                    for visual_idx in range(visual_first):
                        write_image_pair(modulations_v[visual_idx], modulations_v_pred[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)

                   

                visual=False 

        code_mse /= n
        total_pred_mse /= n
        return total_pred_mse, code_mse, detailed_mse

def batch_eval_loop_full_grid(model, inr, loader, full_u_test, full_grid_te, timestamps, detailed_mse, 
                     n, multichannel, z_mean, z_std, dataset_name, 
                     interpolation_seq, n_cond, code_dim=0, visual_first=0, visual_path='', visual_mod=0):
    interpolation_seq = interpolation_seq - n_cond
    visual=True
    # full_grid = shape2coordinates([256,256])
    # full_grid = full_grid[::4,::4]
    # print(full_grid.shape)
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
                # gt = get_reconstructions(
                #     inr, coords, modulations, z_mean, z_std, dataset_name
                # )
                # origin = get_reconstructions(
                #     inr, coords, torch.zeros_like(modulations), torch.zeros_like(z_mean), torch.ones_like(z_std), dataset_name
                # )
                full_images = full_u_test[idx]
                full_grid = full_grid_te[idx].cuda()
                with torch.no_grad():
                    pred_full = get_reconstructions(
                        inr, full_grid, z_pred, z_mean, z_std, dataset_name
                    )

                # if gt.dim() == 5:
                # gt_reshape = gt.permute(0,4,1,2,3).detach().cpu().numpy()
                # origin_reshape = origin.permute(0,4,1,2,3).detach().cpu().numpy()
                pred_reshape = pred_full.permute(0,4,1,2,3).detach().cpu().numpy()
                images_reshape = full_images.permute(0,4,1,2,3).detach().cpu().numpy()
                full_grid_reshape = full_grid.permute(0,4,1,2,3).detach().cpu().numpy()
                print(pred_reshape.shape)
                print(images_reshape.shape)
                # import pdb; pdb.set_trace()
                evmax = np.max(np.abs(pred_reshape[:visual_first,4::5]-images_reshape[:visual_first,4::5]))
                for visual_idx in range(visual_first):
                    # ns
                    # write_prediction(images_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 0, path=visual_path+f'gt_{visual_idx}.pdf', cmap='twilight_shifted', divider=1)
                    # write_prediction(pred_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 0, path=visual_path+f'pred_{visual_idx}.pdf', cmap='twilight_shifted', divider=1)
                    # error = np.abs(images_reshape[visual_idx,4::5] - pred_reshape[visual_idx,4::5])
                    # write_error(error, error, 0, path=visual_path+f'error_{visual_idx}.pdf', cmap='twilight_shifted', divider=1, evmax=evmax)
                    # shallow water
                    write_prediction_sw(full_grid_reshape,images_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 1, path=visual_path+f'gt_{visual_idx}', cmap='twilight_shifted', divider=1)
                    write_prediction_sw(full_grid_reshape,pred_reshape[visual_idx,4::5], images_reshape[visual_idx,4::5], 1, path=visual_path+f'pred_{visual_idx}', cmap='twilight_shifted', divider=1)
                    # error = np.abs(images_reshape[visual_idx,4::5] - pred_reshape[visual_idx,4::5])
                    # write_error(error, error, 0, path=visual_path+f'error_{visual_idx}.pdf', cmap='twilight_shifted', divider=1, evmax=evmax)


                if visual_mod > 0:
                    modulations_v = modulations.permute(0,2,1)
                    modulations_v = modulations_v[...,code_dim:].reshape(modulations_v.shape[0], modulations_v.shape[1], -1, visual_mod, visual_mod)
                    divider = 2 * modulations_v.shape[2]
                    modulations_v = modulations_v.permute(0,2,1,3,4).reshape(modulations_v.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                    modulations_v_pred = z_pred.permute(0,2,1)
                    modulations_v_pred = modulations_v_pred[...,code_dim:].reshape(modulations_v_pred.shape[0], modulations_v_pred.shape[1], -1, visual_mod, visual_mod)
                    modulations_v_pred = modulations_v_pred.permute(0,2,1,3,4).reshape(modulations_v_pred.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                     
                    
                    for visual_idx in range(visual_first):
                        write_image_pair(modulations_v[visual_idx], modulations_v_pred[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)


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
                    modulations_v_pred = z_pred.permute(0,2,1)
                    modulations_v_pred = modulations_v_pred.reshape(modulations_v_pred.shape[0], modulations_v_pred.shape[1], -1, visual_mod, visual_mod)
                    modulations_v_pred = modulations_v_pred.permute(0,2,1,3,4).reshape(modulations_v_pred.shape[0], -1, visual_mod, visual_mod, 1).detach().cpu().numpy()
                     
                    for visual_idx in range(visual_first):
                        write_image_pair(modulations_v[visual_idx], modulations_v_pred[visual_idx], 0, path=visual_path+f'_{visual_idx}_mod.png', cmap='twilight_shifted', divider=divider)

                   

                visual=False 

        code_mse /= n
        total_pred_mse /= n
        return total_pred_mse, code_mse, detailed_mse
