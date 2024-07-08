srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 123 & 
sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 124 & 
sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_inr_ns_dino_v0_r3_mlc.sh 4 0.2 0.2 8 32 1e-2 1e-1 5e-6 125 & 
sleep 5s 