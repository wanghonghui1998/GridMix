srun -p RTX3090 -J wf_r3 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_grid64.sh & 
sleep 5s 

srun -p RTX3090 -J wf_r3 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_grid64_woscale.sh & 
sleep 5s 

srun -p RTX3090 -J wf_r3 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_grid32.sh & 
sleep 5s 

srun -p RTX3090 -J wf_r3 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_grid32_woscale.sh & 
sleep 5s 

