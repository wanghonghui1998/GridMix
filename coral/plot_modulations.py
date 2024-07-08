import numpy as np
import torch
from matplotlib import pyplot as plt 

mod_path = '/cluster/home3/whh/workspace/pinn/exp_nvme4a/wandb/navier-stokes-nms-f40-64-wonorm/modulations/wf_r3_first40_T64-wonorm2.pt'
# mod_path = '/cluster/home3/whh/workspace/pinn/exp_nvme4a/wandb/navier-stokes-nms-40-64-wonorm/modulations/wf_r3_last40_T64-wonorm2.pt'
mod = torch.load(mod_path, map_location='cpu')
z_train_extra = mod['z_train_extra'].numpy()

# print(z_train_extra.shape)  # torch.Size([256, 128, 40])
plt.figure()
x = np.arange(z_train_extra.shape[2])
for i in range(10):
    plt.plot(x, z_train_extra[0, i])
    plt.savefig('/cluster/home1/whh/new_repo/coral/coral/mod_f40.png')
    # import pdb; pdb.set_trace()