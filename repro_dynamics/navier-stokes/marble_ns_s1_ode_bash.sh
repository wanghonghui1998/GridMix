

srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s1_ode.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 123 &
sleep 5s


srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s1_ode.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 124 & 

sleep 5s


srun -p RTX3090 -x node06 -J MoG -n1 --gres gpu:1 bash ./repro_dynamics/navier-stokes/marble_ns_s1_ode.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 125 & 

sleep 5s
