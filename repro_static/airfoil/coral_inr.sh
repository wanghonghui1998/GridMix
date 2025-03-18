export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
seed=$1
name='coral_airfoil_seed'$seed

python3 static/design_inr.py "wandb.name=$name" "data.seed=$seed" "data.dataset_name=airfoil" 'optim.epochs=5000' 'inr_in.w0=5' 'inr_out.w0=15' 'optim.lr_inr=1e-4' 'optim.meta_lr_code=1e-4' 

