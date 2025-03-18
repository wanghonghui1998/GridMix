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
export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
# conda init bash
# conda activate coral
# dir='/cluster/home1/whh/new_repo/DINo/results'
dir='/cluster/data7a/whh/data/results'
dataset_name="navier-stokes-dino"
same_grid=True
sub_from=$1
sub_tr=$2
sub_te=$3
seq_inter_len=20
seq_extra_len=20
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

w0=10
inr_batch_size=64
saved_checkpoint=False

latent_dim=$4
grid_size=$5
grid_base=$6
share_grid=False
grid_sum=False
lr_grid=$7
lr_code=$8
meta_lr_code=$9
seed=${10}

run_name='marble_s1_seed'$seed
name='dyn_marble_s1_seed'$seed
python3 dynamics_modeling/train.py "optim.gamma_step=$gamma_step" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.saved_checkpoint=$saved_checkpoint" # "wandb.checkpoint_path=$checkpoint_path" "wandb.evaluate=$evaluate"
