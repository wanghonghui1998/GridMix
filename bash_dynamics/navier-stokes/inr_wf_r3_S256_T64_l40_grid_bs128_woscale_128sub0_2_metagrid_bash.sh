

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2_metagrid.sh 3072 32 0.8 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2_metagrid.sh 3072 32 0.6 & 
sleep 5s 

srun -p RTX3090 -x node07 -J sub1 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_T64_l40_grid_bs128_woscale_128sub0_2_metagrid.sh 3072 32 0.4 & 
sleep 5s 