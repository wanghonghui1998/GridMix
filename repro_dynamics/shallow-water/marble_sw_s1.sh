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
export PYTHONPATH=/cluster/home1/whh/new_repo/MARBLE_new
# conda init bash
# conda activate coral
dir='/cluster/nvme4a/whh/dataset/shallow_water'
dataset_name='shallow-water-dino'
model_type='siren_GridMix_sw'
data_to_encode=$1 # 'height'
same_grid=True
sub_from=2
sub_tr=$2
sub_te=$3
seq_inter_len=20
seq_extra_len=20
batch_size=16
lr_inr=0.000005
epochs=10000
latent_dim=$4   # 256
depth=6
w0=10
hidden_dim=256
saved_checkpoint=False
use_norm=False
modulate_scale=False
grid_size=$5
grid_size_2=$6
grid_base=$7
share_grid=False
grid_sum=False
lr_grid=$8
lr_code=$9
meta_lr_code=${10}
seed=${11}
launcher='slurm'
port=${12}

name='marble_sw_s1_'$data_to_encode'_seed'$seed
# name='2repro_ddpV2_L'$latent_dim'_MoGChan'$grid_size'x'$grid_size_2'_'$grid_base'_'$share_grid'_'$grid_sum'_'$lr_grid'_alpha'$lr_code'_mlc'$meta_lr_code'_seed'$seed'_sw_dino_'$data_to_encode'_sub'$sub_tr
# id='2repro_ddpV2_L'$latent_dim'_MoGChan'$grid_size'x'$grid_size_2'_'$grid_base'_'$share_grid'_'$grid_sum'_'$lr_grid'_alpha'$lr_code'_mlc'$meta_lr_code'_seed'$seed'_sw_dino_'$data_to_encode'_sub'$sub_tr
python3 inr/inr_ddp.py "data.data_to_encode=$data_to_encode" "ddp.launcher=$launcher" "ddp.port=$port" "data.seed=$seed" "inr.w0=$w0" "optim.meta_lr_code=$meta_lr_code" "optim.lr_code=$lr_code" "inr.grid_sum=$grid_sum" "inr.share_grid=$share_grid" "inr.grid_base=$grid_base" "optim.lr_grid=$lr_grid" "inr.use_norm=$use_norm" "inr.grid_size_2=$grid_size_2" "inr.grid_size=$grid_size" "inr.modulate_scale=$modulate_scale" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" #"wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
# torchrun --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:$port
