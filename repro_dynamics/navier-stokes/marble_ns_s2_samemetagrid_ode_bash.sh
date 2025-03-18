

CUDA_VISIBLE_DEVICES=1, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=0, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=3, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.2 0.2 8 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 

wait 
CUDA_VISIBLE_DEVICES=1, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 123 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=0, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 124 0.4 & 
sleep 5s 

CUDA_VISIBLE_DEVICES=3, bash ./repro_dynamics/navier-stokes/marble_ns_s2_samemetagrid_ode.sh 4 0.05 0.05 2 32 1e-2 1e-1 0.0 125 0.4 & 
sleep 5s 