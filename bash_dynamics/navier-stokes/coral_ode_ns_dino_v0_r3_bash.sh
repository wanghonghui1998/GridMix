srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3.sh 4 0.2 0.2 64 124 & 
sleep 5s 

srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3.sh 4 0.2 0.2 64 125 & 
sleep 5s 

srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3.sh 4 0.2 0.2 128 123 & 
sleep 5s 

srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3.sh 4 0.2 0.2 128 125 & 
sleep 5s 

srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_nms_v0_r3.sh 4 0.2 0.2 64 123 & 
sleep 5s 

srun -p RTX3090 -J dyn -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_nms_v0_r3.sh 4 0.2 0.2 64 125 & 
sleep 5s 