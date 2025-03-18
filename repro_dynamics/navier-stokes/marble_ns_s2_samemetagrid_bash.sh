

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 


srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 