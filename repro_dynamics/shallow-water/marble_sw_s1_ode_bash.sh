
srun -p A100 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s1_ode.sh 1 1 256 8 16 32 1e-2 1e-1 0.0 123 & 
sleep 5s 

srun -p A100 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s1_ode.sh 1 1 256 8 16 32 1e-2 1e-1 0.0 124 & 
sleep 5s 

srun -p A100 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s1_ode.sh 1 1 256 8 16 32 1e-2 1e-1 0.0 125 & 
sleep 5s 
