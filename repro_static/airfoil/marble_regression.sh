export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
latent_dim=$1
grid_size=$2
grid_size_2=$3
grid_base=$4
lr_grid=$5
lr_code=$6  # 1e-2
meta_lr_code=$7 # 1e-4
seed=$8
optim_in=$9
optim_out=${10} 
w0_in=${11} # 5
w0_out=${12}    # 10

model_type='siren_GridMix'
use_norm=False 
siren_init=True
grid_sum=False
share_grid=False

run_name_in='coral_airfoil_seed'$seed
run_name='marble_airfoil_seed'$seed
name='reg_marble_airfoil_seed'$seed

python3 static/design_regression.py "inr.run_name_in=$run_name_in" "wandb.name=$name" "data.dataset_name=airfoil" "inr.run_name=$run_name" 'optim.epochs=10000' # 'model.width=128' 'model.depth=3' 'inr.inner_steps=3' 
