srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_sst_S202_T22.sh 3 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_sst_S202_T22.sh 4 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_sst_S202_T22.sh 8 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_sst_S202_T22.sh 9 & 
sleep 5s 


