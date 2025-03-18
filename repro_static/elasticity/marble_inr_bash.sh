srun -p A100 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/marble_inr.sh 128 8 64 1e-2 1e-2 1e-4 123 False True & 
sleep 5s 

srun -p A100 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/marble_inr.sh 128 8 64 1e-2 1e-2 1e-4 124 False True & 
sleep 5s 


srun -p A100 -J elasticity -n1 --gres gpu:1 bash ./repro_static/elasticity/marble_inr.sh 128 8 64 1e-2 1e-2 1e-4 125 False True & 
sleep 5s 