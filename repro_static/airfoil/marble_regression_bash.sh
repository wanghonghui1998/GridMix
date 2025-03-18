

srun -p A100 -J airfoil -n1 --gres gpu:1 bash ./repro_static/airfoil/marble_regression.sh 128 32 8 32 1e-1 1e-2 1e-4 123 False True 5 15 & 
sleep 5s 


srun -p A100 -J airfoil -n1 --gres gpu:1 bash ./repro_static/airfoil/marble_regression.sh 128 32 8 32 1e-1 1e-2 1e-4 124 False True 5 15 & 
sleep 5s 


srun -p A100 -J airfoil -n1 --gres gpu:1 bash ./repro_static/airfoil/marble_regression.sh 128 32 8 32 1e-1 1e-2 1e-4 125 False True 5 15 & 
sleep 5s 

