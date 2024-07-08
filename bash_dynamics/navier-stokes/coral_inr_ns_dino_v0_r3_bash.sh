srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 1 4 4 &
sleep 5s 

srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 1 2 2 &
sleep 5s 

srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 1 0.0125 0.0125 &
sleep 5s 

srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 1 0.05 0.05 &
sleep 5s 


srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 4 0.2 0.2 &
sleep 5s 

srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 2 0.2 0.2 &
sleep 5s 