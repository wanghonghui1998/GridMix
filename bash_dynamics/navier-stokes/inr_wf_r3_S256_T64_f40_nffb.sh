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
dir='/cluster/nvme4a/whh/dataset/ns_nms/u_Re800_T60_transient_tr256.npy'
dataset_name='navier-stokes-nms-f40-64-wonorm'
model_type='nffb'
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
seq_inter_len=20
seq_extra_len=20
batch_size=128
lr_inr=0.000005
epochs=10000
latent_dim=128
depth=4
hidden_dim=128
saved_checkpoint=False
n_levels=3
base_resolution=8
per_level_scale=2
n_features_per_level=2
base_sigma=5.0

name='nffb'$base_sigma'_'$n_levels'_'$base_resolution'_'$per_level_scale'_F'$n_features_per_level'_wf_r3_first40_T64-wonorm2'
id='nffb'$base_sigma'_'$n_levels'_'$base_resolution'_'$per_level_scale'_F'$n_features_per_level'_wf_r3_S256_first40_T64-wonorm2'
python3 inr/inr.py "inr.base_sigma=$base_sigma" "inr.n_levels=$n_levels" "inr.n_features_per_level=$n_features_per_level" "inr.per_level_scale=$per_level_scale" "inr.base_resolution=$base_resolution" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
