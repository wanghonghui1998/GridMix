export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
latent_dim=$1
grid_size=$2
grid_base=$3
lr_grid=$4
lr_code=$5  # 1e-2
meta_lr_code=$6 # 1e-4
seed=$7
optim_in=$8
optim_out=$9 

model_type='siren_GridMix'
use_norm=False 
siren_init=True
grid_sum=False
share_grid=False

run_name_in='coral_elasticity_seed'$seed
run_name='marble_elasticity_seed'$seed
name='reg_marble_elasticity_seed'$seed

python3 static/design_regression.py "inr.run_name_in=$run_name_in" "wandb.name=$name" "data.dataset_name=elasticity" "inr.run_name=$run_name" 'optim.epochs=10000'
