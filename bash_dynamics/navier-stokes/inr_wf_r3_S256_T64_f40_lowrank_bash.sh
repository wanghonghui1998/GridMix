srun -p RTX3090 -x node07 -J lr1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_lowrank.sh 512 1 & 
sleep 5s 

srun -p RTX3090 -x node07 -J lr2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_lowrank.sh 1024 2 & 
sleep 5s 

srun -p RTX3090 -x node07 -J lr4 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_lowrank.sh 2048 4 & 
sleep 5s 

srun -p RTX3090 -x node07 -J lr8 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_f40_lowrank.sh 4096 8 & 
sleep 5s 

