srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_40.sh &

sleep 5s 

srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r4_40.sh &

sleep 5s 

srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r5_40.sh &

sleep 5s 

srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wof_r2_40.sh &

sleep 5s 


srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wof_r3_40.sh &

sleep 5s 


srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wof_r4_40.sh &

sleep 5s 

