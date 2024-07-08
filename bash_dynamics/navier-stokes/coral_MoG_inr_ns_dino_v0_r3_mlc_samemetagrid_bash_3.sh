srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 16 32 1e-2 1e-1 5e-6 123 0.4 & 
sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 16 32 1e-2 1e-1 5e-6 124 0.4 & 
sleep 5s 

# CUDA_VISIBLE_DEVICES=2, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 125 0.4 & 
# sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 16 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 16 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

# CUDA_VISIBLE_DEVICES=6, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 125 0.4 & 
# sleep 5s 