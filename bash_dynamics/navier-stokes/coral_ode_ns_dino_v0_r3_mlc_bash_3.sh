# srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_inr_ns_dino_v0_r3.sh 1 4 4 &
# sleep 5s 


srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3_mlc.sh 1 4 4 64 10 123 0.0 &
sleep 5s 


srun -p A100 -w node09 -J dino -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/coral_ode_ns_dino_v0_r3_mlc.sh 1 4 4 64 10 126 0.0 &
sleep 5s 