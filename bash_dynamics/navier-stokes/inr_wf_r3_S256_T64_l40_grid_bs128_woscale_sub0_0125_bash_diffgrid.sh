srun -p RTX3090 -x node07 -J sub2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_128sub0_2_wnorm.sh &
sleep 5s 

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2.sh 12288 64 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2.sh 3072 32 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2.sh 768 16 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_0.0125.sh & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_sub0_0125.sh 12288 64 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_sub0_0125.sh 3072 32 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub2 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_sub0_0125.sh 768 16 & 
sleep 5s 
