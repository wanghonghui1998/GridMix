srun -p RTX3090 -J enc -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_dino_enc.sh 1e-3 &
sleep 5s 

srun -p RTX3090 -J enc -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_dino_enc.sh 1e-4 & 
sleep 5s 

srun -p RTX3090 -J enc -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_dino_enc.sh 1e-5 & 
sleep 5s 
