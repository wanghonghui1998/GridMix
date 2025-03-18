srun -p RTX3090 -x node06 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/coral_inr.sh 123 & 
sleep 5s 

srun -p RTX3090 -x node06 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/coral_inr.sh 124 & 
sleep 5s 

srun -p RTX3090 -x node06 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/coral_inr.sh 125 & 
sleep 5s 
