srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_enc_lrcode.sh 1e-3 
sleep 5s 


# srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_enc_lrcode.sh 1e-4 &
# sleep 5s 

# srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_enc_lrcode.sh 1e-5 &
# sleep 5s 


# srun -p RTX3090 -J coral -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_dino_enc_lrcode.sh 1e-3 
# sleep 5s 
