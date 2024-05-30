#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=ode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

# source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
# conda init bash
# conda activate coral
zone_id=$1
dir='/cluster/nvme4a/whh/dataset/sst/data_zone_'$zone_id'.h5'
dataset_name="sst-11-22"
same_grid=True
sub_from=1
sub_tr=1
sub_te=1
seq_inter_len=11
seq_extra_len=11
batch_size=32

epochs=10000
lr=0.001
weight_decay=0
gamma_step=0.75

depth=3
width=512

teacher_forcing_init=0.99
teacher_forcing_decay=0.99
teacher_forcing_update=10
inner_steps=3

saved_checkpoint=True
evaluate=True 
checkpoint_path='/cluster/home1/whh/workspace/wandb/sst-11-22/model/dynfno_bs32_sst_zone_'$zone_id'.pt'

#run_name="toasty-darkness-5007"  #"desert-sponge-4958" # "eager-field-4969" # splendid-yogurt-4959 # "desert-sponge-4958" # misunderstood-sunset-4916
run_name='sst_zone_'$zone_id
name='dynfno_bs32_sst_zone_evaluate_'$zone_id
id='dynfno_bs32_sst_S202_zone_bs64_evaluate_best_'$zone_id
python3 dynamics_modeling/train_fno_ode.py "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.id=$id" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.checkpoint_path=$checkpoint_path" "wandb.evaluate=$evaluate"
