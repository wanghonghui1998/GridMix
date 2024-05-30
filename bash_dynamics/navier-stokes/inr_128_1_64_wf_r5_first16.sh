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
dir='/cluster/home1/whh/workspace/Neural-Spectral-Methods/src/pde/navierstokes/u_Re80000_T60_transient.npy'
dataset_name='navier-stokes-nms'
model_type='siren'
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
seq_inter_len=16
seq_extra_len=48
batch_size=16
lr_inr=0.000005
epochs=20000
latent_dim=128
depth=4
hidden_dim=128
saved_checkpoint=False
inner_steps=3
test_inner_steps=3
name='w_forcing_r5_1_first16_1_64'

python3 inr/inr.py "optim.inner_steps=$inner_steps" "optim.test_inner_steps=$test_inner_steps" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" # "wandb.checkpoint_path=$checkpoint_path" #"wandb.id=$id" "wandb.dir=$dir" 
