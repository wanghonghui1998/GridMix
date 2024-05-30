srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_mfn.sh 9 & 
sleep 5s 


srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_ff.sh 9 & 
sleep 5s 

srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_sst_S202_T22_bacon.sh 9 & 
sleep 5s 


srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_ff.sh & 
sleep 5s 

# srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_bacon.sh & 
# sleep 5s 

# srun -p RTX3090 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/ode_wf_r3_f40_wonorm_mfn.sh & 
# sleep 5s 
