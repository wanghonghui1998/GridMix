CUDA_VISIBLE_DEVICES=0, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 123 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=1, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 124 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=2, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 125 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=3, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=4, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=5, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 