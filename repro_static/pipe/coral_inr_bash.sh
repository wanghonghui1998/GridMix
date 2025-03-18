srun -p RTX3090 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/coral_inr.sh 123 & 
sleep 5s 

srun -p RTX3090 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/coral_inr.sh 124 & 
sleep 5s 

srun -p RTX3090 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/coral_inr.sh 125 & 
sleep 5s 
