srun -p RTX3090 -x node07 -J gridlr1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_grid64_bs128_woscale_lowrank.sh 384 1 & 
sleep 5s 

# srun -p RTX3090 -x node07 -J gridlr2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_grid64_bs128_woscale_lowrank.sh 768 2 & 
# sleep 5s 

srun -p RTX3090 -x node07 -J gridlr4 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_grid64_bs128_woscale_lowrank.sh 1536 4 & 
sleep 5s 

# srun -p RTX3090 -x node07 -J gridlr8 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_grid64_bs128_woscale_lowrank.sh 3072 8 & 
# sleep 5s 

srun -p RTX3090 -x node07 -J gridlr16 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_grid64_bs128_woscale_lowrank.sh 6144 16 & 
sleep 5s 

