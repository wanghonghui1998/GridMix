CUDA_VISIBLE_DEVICES=3, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 123 5e-6 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=4, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 124 5e-6 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=5, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 125 5e-6 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=6, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 123 0.0 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=3, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 124 0.0 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=4, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3_mlc_samemetagrid.sh 4 0.05 0.05 64 10 125 0.0 0.4 & 
sleep 5s 
