CUDA_VISIBLE_DEVICES=7, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/memory_hippo_wf_r3_40_wonorm.sh 128 1 & 

sleep 5s 

CUDA_VISIBLE_DEVICES=6,  bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/memory_hippo_wf_r3_40_wonorm.sh 64 2 & 

sleep 5s 
CUDA_VISIBLE_DEVICES=5, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/memory_hippo_wf_r3_40_wonorm.sh 32 4 & 

sleep 5s 
CUDA_VISIBLE_DEVICES=4, bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/memory_hippo_wf_r3_40_wonorm.sh 16 8 & 

sleep 5s 
