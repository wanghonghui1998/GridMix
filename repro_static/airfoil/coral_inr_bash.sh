srun -p A100 -J naca-euler -n1 --gres gpu:1 bash ./repro_static/airfoil/coral_inr.sh 123 & 
sleep 5s 

srun -p A100 -J naca-euler -n1 --gres gpu:1 bash ./repro_static/airfoil/coral_inr.sh 124 & 
sleep 5s 

srun -p A100 -J naca-euler -n1 --gres gpu:1 bash ./repro_static/airfoil/coral_inr.sh 125 & 
sleep 5s 
