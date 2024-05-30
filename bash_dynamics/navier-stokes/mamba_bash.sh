srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/mamba_wf_r3_40.sh 8 &

sleep 5s 

srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/mamba_wf_r3_40.sh 1 &

sleep 5s 

srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/mamba_wf_r3_40.sh 4 &

sleep 5s 


srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/mamba_wf_r3_40.sh 2 &

sleep 5s 


srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/mamba_wf_r3_40.sh 32 &

sleep 5s 