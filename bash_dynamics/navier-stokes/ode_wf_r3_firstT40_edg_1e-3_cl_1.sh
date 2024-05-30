# srun -p RTX3090 -x node06 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-3_cl.sh 1.0 1.0 & 

# sleep 5s 

# srun -p RTX3090 -x node06 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-3_cl.sh 0.1 1.0 & 

# sleep 5s 


srun -p RTX3090 -w master -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-3_cl.sh 1.0 0.1 & 

sleep 5s 


srun -p RTX3090 -w master -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_firstT40_edg_1e-3_cl.sh 0.1 0.1 & 

sleep 5s 

