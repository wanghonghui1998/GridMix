

srun -p A100 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/marble_inr.sh 128 8 32 1e-1 2e-1 5e-5 123 False True 5 10 & 
sleep 5s 

srun -p A100 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/marble_inr.sh 128 8 32 1e-1 2e-1 5e-5 124 False True 5 10 & 
sleep 5s 

srun -p A100 -J pipe -n1 --gres gpu:1 bash ./repro_static/pipe/marble_inr.sh 128 8 32 1e-1 2e-1 5e-5 125 False True 5 10 & 
sleep 5s 
