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
w0_out=${12}    # 15

launcher='slurm'
port=${13}

model_type='siren_GridMix'
use_norm=False 
siren_init=True
grid_sum=False
share_grid=False

name='marble_airfoil_seed'$seed

python3 static/design_inr_ddp.py "ddp.launcher=$launcher" "ddp.port=$port" "optim.optim_in=$optim_in" "optim.optim_out=$optim_out" "data.seed=$seed" "data.dataset_name=airfoil" 'optim.batch_size=32' 'optim.epochs=5000' "inr_in.model_type=$model_type" "inr_out.model_type=$model_type" "inr_in.latent_dim=$latent_dim" "inr_out.latent_dim=$latent_dim" "inr_in.grid_size=$grid_size" "inr_out.grid_size=$grid_size" "inr_in.grid_size_2=$grid_size_2" "inr_out.grid_size_2=$grid_size_2" "inr_in.grid_base=$grid_base" "inr_out.grid_base=$grid_base" "inr_in.use_norm=$use_norm" "inr_out.use_norm=$use_norm" "inr_in.siren_init=$siren_init" "inr_out.siren_init=$siren_init" "inr_in.grid_sum=$grid_sum" "inr_out.grid_sum=$grid_sum" "inr_in.share_grid=$share_grid" "inr_out.share_grid=$share_grid" "inr_in.w0=$w0_in" "inr_out.w0=$w0_out" 'optim.lr_inr=1e-4' "optim.lr_grid=$lr_grid" "optim.lr_code=$lr_code" "optim.meta_lr_code=$meta_lr_code" "wandb.name=$name" 

