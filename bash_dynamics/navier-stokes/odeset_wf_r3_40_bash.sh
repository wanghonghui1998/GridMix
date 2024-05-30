srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_wf_r3_40_wonorm.sh & 
sleep 5s 

srun -p RTX3090 -x node07 -J sst -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_wf_r3_40_128sub0_2_wonorm.sh & 
sleep 5s 

srun -p RTX3090 -x node07 -n1 --gres gpu:1 bash /cluster/home1/whh/new_repo/coral/bash_dynamics/navier-stokes/odeset_wf_r3_40_sub0_0125_wonorm.sh & 
sleep 5s 

