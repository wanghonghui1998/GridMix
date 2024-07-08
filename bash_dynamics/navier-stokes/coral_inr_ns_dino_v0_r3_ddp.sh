#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

# source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
# conda init bash
# conda activate coral
dir='/cluster/home1/whh/new_repo/DINo/results'
dataset_name='navier-stokes-dino'
model_type='siren'
same_grid=True
sub_from=$1
sub_tr=$2
sub_te=$3
seq_inter_len=20
seq_extra_len=20
batch_size=128
lr_inr=0.000005
epochs=10000
latent_dim=128
depth=4
hidden_dim=128
saved_checkpoint=False

launcher='slurm'
port=$4

name='ddp_coral_ns_dino_v0_r3_256sub'$1'_'$2'_'$3
id='ddp_coral_ns_dino_v0_r3_256sub'$1'_'$2'_'$3
python3 inr/inr_ddp.py "ddp.launcher=$launcher" "ddp.port=$port" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
