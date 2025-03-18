


srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh vorticity 1 1 256 8 16 32 1e-2 1e-1 0.0 123 29297 & 
sleep 5s 

srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh height 1 1 256 8 16 32 1e-2 1e-1 0.0 123 29296 & 
sleep 5s 

srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh vorticity 1 1 256 8 16 32 1e-2 1e-1 0.0 124 29397 & 
sleep 5s 

srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh height 1 1 256 8 16 32 1e-2 1e-1 0.0 124 29396 & 
sleep 5s 

srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh vorticity 1 1 256 8 16 32 1e-2 1e-1 0.0 125 29387 & 
sleep 5s 

srun -p A100 -w node09 -J sw -n4 --gres gpu:4 bash ./repro_dynamics/shallow-water/marble_sw_s1.sh height 1 1 256 8 16 32 1e-2 1e-1 0.0 125 29386 & 
sleep 5s 





