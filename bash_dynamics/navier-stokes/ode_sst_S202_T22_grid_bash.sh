srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_grid64.sh 4 & 
sleep 5s 

srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_grid64.sh 9 & 
sleep 5s 

srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_grid32.sh 4 & 
sleep 5s 

srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_grid32.sh 9 & 
sleep 5s 


