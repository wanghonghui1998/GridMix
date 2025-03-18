export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
seed=$1
name='coral_pipe_seed'$seed

python3 static/design_inr.py "data.seed=$seed" "data.dataset_name=pipe" 'optim.batch_size=16' 'optim.epochs=5000' 'inr_in.w0=5' 'inr_out.w0=10' 'inr_in.hidden_dim=128' 'inr_in.depth=5' 'inr_out.hidden_dim=128' 'inr_out.depth=5' 'optim.lr_inr=5e-5' 'optim.meta_lr_code=5e-5' "wandb.name=$name" 

