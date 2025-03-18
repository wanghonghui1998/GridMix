
srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 




srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 




srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.2 0.2 256 8 16 32 1e-2 1e-1 0.0 129 0.4 & 
sleep 5s 





srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 






srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 




srun -p RTX3090 -J sw -n1 --gres gpu:1 bash ./repro_dynamics/shallow-water/marble_sw_s2_samemetagrid_ode.sh 0.05 0.05 256 4 8 32 1e-2 1e-1 0.0 126 0.4 & 
sleep 5s 


