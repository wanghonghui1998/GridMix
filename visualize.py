"""
Visualize outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import matplotlib.gridspec as gridspec

def write_1d(gt, pred, path):
    """
    create a figure with 2 subplots, one for the gt and one for the pred
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    # implot gt and pred with shape(t,x) as images, use gt range to normalize the color
    gt_range = [np.min(gt), np.max(gt)]
    axs[0].imshow(gt, aspect='auto', origin='lower', cmap='viridis', vmin=gt_range[0], vmax=gt_range[1])
    axs[0].set_title('Ground Truth')
    # axs[0].axis('off')
    axs[1].imshow(pred, aspect='auto', origin='lower', cmap='viridis', vmin=gt_range[0], vmax=gt_range[1])
    axs[1].set_title('Prediction')
    # axs[1].axis('off')
    plt.savefig(path, dpi=72, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig.clf()

def write_prediction_sw(index, batch_gt, batch_gt_range, state_idx, path, cmap='plasma', divider=1):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """

    N, K = 15, 4 # remove dark colors
    # cmap = 'twilight_shifted'
    cmap = plt.get_cmap("RdBu_r") # RdBu_r

    seq_len, height, width, state_c = batch_gt.shape  # [20, 64, 64, 1]
    # t_horizon = seq_len // divider
    # new_seq_len = t_horizon * divider 
    # batch_gt = batch_gt[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    clim = np.max(np.abs(batch_gt_range[...,state_idx]))
    x = index[0,0,:,:,0] #.reshape(-1,1)
    y = index[0,0,:,:,1] #.reshape(-1,1)
    z = index[0,0,:,:,2] #.reshape(-1,1)
    print(z.shape)
    

    # import pdb; pdb.set_trace()
    for i in range(seq_len):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.view_init(elev=30, azim=120, roll=0)
        # clim = np.max(np.abs(data))
        norm = clr.Normalize(-clim, clim)
        fc = cmap(norm(batch_gt[i,:,:,state_idx]))
        surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
        ax.set_box_aspect((1,1,1))
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)
        ax.axis('off')
            
        plt.savefig(path+f'{i}.pdf', dpi=50, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        fig.clf()
        plt.close(fig)

def write_error(batch_gt, batch_gt_range, state_idx, path, cmap='plasma', divider=1, evmax=1.0):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """

    # N, K = 15, 4 # remove dark colors
    # cmap = 'twilight_shifted'

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = clr.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    

    seq_len, height, width, state_c = batch_gt.shape  # [20, 64, 64, 1]
    t_horizon = seq_len // divider
    new_seq_len = t_horizon * divider 
    batch_gt = batch_gt[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    fig = plt.figure(figsize=((t_horizon+1)*6, divider*6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(divider, t_horizon),  # creates 2x2 grid of axes
                     axes_pad=0.05, # pad between axes in inch.
                     share_all=True,
                     cbar_location="right",
                     cbar_mode='edge',
                     direction = 'row',
                     cbar_size="10%",
                     cbar_pad=0.15)  
    vmax = np.max(batch_gt_range[...,state_idx])
    vmin = np.min(batch_gt_range[...,state_idx])

    cmap = plt.get_cmap("twilight_shifted")
    cmap = truncate_colormap(cmap, 0.5, 0.8)
    print(evmax)
    norm = clr.BoundaryNorm(np.linspace(0, evmax, 11), 256)
    # vmin = vmin * (N + 2*K) / N
    # vmax = vmax * (N + 2*K) / N
    # norm = clr.BoundaryNorm(np.linspace(vmin, vmax, N+2*K), cmap.N)


    for traj in range(divider):
        for t in range(t_horizon):
            # Iterating over the grid returns the Axes.
            im = grid[traj * t_horizon + t].imshow(batch_gt[traj, t, :, :, state_idx], norm=norm, cmap=cmap, interpolation='none')
            grid[traj * t_horizon + t].set_axis_off()
            if t == t_horizon-1:
                cb=grid[traj * t_horizon + t].cax.colorbar(im)
                # cb.ax.set_yticklabels(["{:.1f}".format(i) for i in cb.get_ticks()])
           
    plt.savefig(path, dpi=72, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    fig.clf()
    plt.close(fig)


def write_prediction(batch_gt, batch_gt_range, state_idx, path, cmap='plasma', divider=1):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """

    N, K = 15, 4 # remove dark colors
    # cmap = 'twilight_shifted'
    cmap = plt.get_cmap("twilight_shifted")

    seq_len, height, width, state_c = batch_gt.shape  # [20, 64, 64, 1]
    t_horizon = seq_len // divider
    new_seq_len = t_horizon * divider 
    batch_gt = batch_gt[:new_seq_len].reshape(divider, t_horizon, height, width, state_c)
    fig = plt.figure(figsize=((t_horizon+1)*6, divider*6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(divider, t_horizon),  # creates 2x2 grid of axes
                     axes_pad=0.05, # pad between axes in inch.
                     share_all=True,
                     cbar_location="right",
                     cbar_mode='edge',
                     direction = 'row',
                     cbar_size="10%",
                     cbar_pad=0.15)  
    vmax = np.max(batch_gt_range[...,state_idx])
    vmin = np.min(batch_gt_range[...,state_idx])

    vmin = vmin * (N + 2*K) / N
    vmax = vmax * (N + 2*K) / N
    norm = clr.BoundaryNorm(np.linspace(vmin, vmax, N+2*K), cmap.N)


    for traj in range(divider):
        for t in range(t_horizon):
            # Iterating over the grid returns the Axes.
            im = grid[traj * t_horizon + t].imshow(batch_gt[traj, t, :, :, state_idx], norm=norm, cmap=cmap, interpolation='none')
            grid[traj * t_horizon + t].set_axis_off()
            if t == t_horizon-1:
                cb=grid[traj * t_horizon + t].cax.colorbar(im)
                # cb.ax.set_yticklabels(["{:.1f}".format(i) for i in cb.get_ticks()])
           
    plt.savefig(path, dpi=72, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    fig.clf()
    plt.close(fig)


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

