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
model_type='siren_MoG_chan'
same_grid=True
sub_from=$1
sub_tr=$2
sub_te=$3
seq_inter_len=20
seq_extra_len=20
batch_size=64
lr_inr=0.000005
epochs=10000
latent_dim=128
depth=4
w0=10
hidden_dim=128
saved_checkpoint=False
use_norm=False
modulate_scale=True
grid_size=$4
grid_base=$5
share_grid=False
grid_sum=False
lr_grid=$6
lr_code=$7
meta_lr_code=$8
seed=$9
grid_ratio=${10}
meta_same_grid=True
launcher=''
port=29051


name='3metaSG'$grid_ratio'_MoGChanS+S'$grid_size'_'$grid_base'_'$share_grid'_'$grid_sum'_'$lr_grid'_alpha'$lr_code'_mlc'$meta_lr_code'_seed'$seed'_256sub'$1'_'$2'_'$3
id='3metaSG'$grid_ratio'_MoGChanS+S'$grid_size'_'$grid_base'_'$share_grid'_'$grid_sum'_'$lr_grid'_alpha'$lr_code'_mlc'$meta_lr_code'_seed'$seed'_256sub'$1'_'$2'_'$3
python3 inr/inr_metagrid.py "optim.meta_same_grid=$meta_same_grid" "optim.grid_ratio=$grid_ratio" "data.seed=$seed" "inr.w0=$w0" "optim.meta_lr_code=$meta_lr_code" "optim.lr_code=$lr_code" "inr.grid_sum=$grid_sum" "inr.share_grid=$share_grid" "inr.grid_base=$grid_base" "optim.lr_grid=$lr_grid" "inr.use_norm=$use_norm" "inr.grid_size=$grid_size" "inr.modulate_scale=$modulate_scale" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
# torchrun --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:$port
