export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
seed=$1
name='coral_elasticity_seed'$seed

python3 static/design_inr.py "data.seed=$seed" "data.dataset_name=elasticity" 'optim.batch_size=64' 'optim.epochs=5000' 'inr_in.w0=10' 'inr_out.w0=15' 'optim.lr_inr=1e-4' 'optim.meta_lr_code=1e-4'  "wandb.name=$name" 

