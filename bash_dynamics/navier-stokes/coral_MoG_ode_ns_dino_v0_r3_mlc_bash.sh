srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc.sh 1 4 4 8 32 1e-2 1e-1 0.0 123 & 
sleep 5s 

srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc.sh 1 4 4 8 32 1e-2 1e-2 0.0 123 & 
sleep 5s 

# srun -p RTX3090 -J MoG -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_MoG_ode_ns_dino_v0_r3_mlc.sh 1 4 4 8 32 1e-2 2e-1 0.0 123 & 
# sleep 5s 