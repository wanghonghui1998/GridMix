"""
Visualize outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import matplotlib.gridspec as gridspec

def write_image(batch_gt, batch_pred, state_idx, path, cmap='plasma', divider=1):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """
    seq_len, height, width, state_c = batch_gt.shape  # [20, 64, 64, 1]
    t_horizon = seq_len // divider
    new_seq_len = t_horizon * divider 
    batch_gt = batch_gt[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    fig = plt.figure(figsize=(t_horizon+1, divider))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(divider, t_horizon),  # creates 2x2 grid of axes
                     axes_pad=0.05, # pad between axes in inch.
                     share_all=True,
                     cbar_location="right",
                     cbar_mode='edge',
                     direction = 'row',
                     cbar_size="10%",
                     cbar_pad=0.15)  
    vmax = np.max(batch_pred[...,state_idx])
    vmin = np.min(batch_pred[...,state_idx])
    for traj in range(divider):
        for t in range(t_horizon):
            # Iterating over the grid returns the Axes.
            im = grid[traj * t_horizon + t].imshow(batch_gt[traj, t, :, :, state_idx], vmax=vmax, vmin=vmin, cmap=cmap, interpolation='none')
            grid[traj * t_horizon + t].set_axis_off()
            if t == t_horizon-1:
                grid[traj * t_horizon + t].cax.colorbar(im)
           
    plt.savefig(path, dpi=72, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    fig.clf()
    plt.close(fig)

def write_image_pair(batch_gt, batch_pred, state_idx, path, cmap='plasma', divider=1, normalize=False):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """
    seq_len, height, width, state_c = batch_gt.shape  # [20, 64, 64, 1]
    t_horizon = seq_len // divider
    new_seq_len = t_horizon * divider 
    batch_gt = batch_gt[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    batch_pred = batch_pred[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    fig = plt.figure(figsize=(t_horizon+1, divider*2))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(divider*2, t_horizon),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     share_all=True,
                     cbar_location="right",
                     cbar_mode='edge',
                     direction = 'row',
                     cbar_size="10%",
                     cbar_pad=0.15)  
    vmax = np.max(batch_gt)
    vmin = np.min(batch_gt)
    if normalize:
        batch_pred = batch_pred - np.mean(batch_pred) + np.mean(batch_gt)
    for traj in range(divider):
        for t in range(t_horizon):
            # Iterating over the grid returns the Axes.
            im = grid[2 * traj * t_horizon + t].imshow(batch_gt[traj, t, :, :, state_idx], vmax=vmax, vmin=vmin, cmap=cmap, interpolation='none')
            im2 = grid[(2 * traj + 1) * t_horizon + t].imshow(batch_pred[traj, t, :, :, state_idx], vmax=vmax, vmin=vmin, cmap=cmap, interpolation='none')
            grid[2 * traj * t_horizon + t].set_axis_off()
            grid[(2 * traj + 1) * t_horizon + t].set_axis_off()
            if t == t_horizon-1:
                grid[2 * traj * t_horizon + t].cax.colorbar(im)
                grid[(2 * traj + 1) * t_horizon + t].cax.colorbar(im2)

    plt.savefig(path, dpi=72, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    fig.clf()
    plt.close(fig)

