
srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 



srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 129 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 129 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh height 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 


srun -p RTX3090 -w node06 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid.sh vorticity 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 
