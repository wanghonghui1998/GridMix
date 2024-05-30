# srun -p RTX3090 -w master -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odefno_sst_S202_T22.sh 3 & 
# sleep 5s 

srun -p RTX3090 -w master -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odefno_sst_S202_T22.sh 4 & 
sleep 5s 

srun -p RTX3090 -w master -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odefno_sst_S202_T22.sh 8 & 
sleep 5s 

srun -p RTX3090 -w master -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odefno_sst_S202_T22.sh 9 & 
sleep 5s 


