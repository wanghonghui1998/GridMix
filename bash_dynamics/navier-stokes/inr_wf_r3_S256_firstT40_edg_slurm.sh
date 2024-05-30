# srun -p A100 -J edg -n2 --gres gpu:2 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_edg.sh & 
# sleep 5s 
# srun -p A100 -J edg -n2 --gres gpu:2 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_edg_1e-3.sh & 
# sleep 5s 
srun -p RTX3090 -J edg -n2 --gres gpu:2 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_edg_1e-4.sh & 
sleep 5s 
srun -p RTX3090 -J edg -n2 --gres gpu:2 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/inr_wf_r3_S256_firstT40_edg_1e-5.sh & 

sleep 5s 
