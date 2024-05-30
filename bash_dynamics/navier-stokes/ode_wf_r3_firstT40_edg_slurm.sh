
# srun -p RTX3090 -J edg -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-2.sh & 
# sleep 5s 

# srun -p RTX3090 -J edg -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-3.sh & 
# sleep 5s 

srun -p RTX3090 -x node06 -J edg -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-4.sh & 
sleep 5s 

srun -p RTX3090 -x node06 -J edg -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-5.sh & 
sleep 5s 

