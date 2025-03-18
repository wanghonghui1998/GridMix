

srun -p A100 -x node04 -J MoG -n4 --gres gpu:4 bash ./repro_dynamics/navier-stokes/marble_ns_s1.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 123 29152 & 

sleep 5s

srun -p A100 -x node04 -J MoG -n4 --gres gpu:4 bash ./repro_dynamics/navier-stokes/marble_ns_s1.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 124 29153 & 

sleep 5s


srun -p A100 -x node04 -J MoG -n4 --gres gpu:4 bash ./repro_dynamics/navier-stokes/marble_ns_s1.sh 1 4 4 32 8 32 1e-2 1e-1 5e-6 125 29154 & 

sleep 5s



