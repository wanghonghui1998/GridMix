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
zone_id=$1
dir='/cluster/nvme4a/whh/dataset/sst/data_zone_'$zone_id'.h5'
dataset_name='sst-11-22'
model_type='siren_grid'
same_grid=True
sub_from=1
sub_tr=1
sub_te=1
seq_inter_len=11
seq_extra_len=11
batch_size=128
lr_inr=0.000005
epochs=10000
latent_dim=6144
depth=4
hidden_dim=128
saved_checkpoint=False

use_norm=False
grid_size=32
modulate_scale=True

name='grid32_V0_2_bs128_sst_zone_'$zone_id
id='grid32_V0_2_bs128_sst_S202_zone_'$zone_id
python3 inr/inr.py "inr.use_norm=$use_norm" "inr.grid_size=$grid_size" "inr.modulate_scale=$modulate_scale" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
