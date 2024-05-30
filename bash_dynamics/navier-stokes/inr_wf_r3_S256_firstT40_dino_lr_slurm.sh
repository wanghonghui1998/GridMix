srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_lrcode1e-3.sh & 
sleep 5s 
srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_lrcode1e-4.sh & 
sleep 5s 
srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_lrcode1e-5.sh & 
sleep 5s 
srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_lrcode1e-6.sh & 

sleep 5s 