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
dataset_name='navier-stokes-nms-40-64-wonorm'
model_type='siren'
same_grid=True
sub_from=2
sub_tr=0.2
sub_te=0.2
seq_inter_len=20
seq_extra_len=20
batch_size=128
lr_inr=0.000005
epochs=10000
latent_dim=128
depth=4
hidden_dim=128
grid_ratio=0.4
saved_checkpoint=True 
checkpoint_path='/cluster/home3/whh/workspace/pinn/exp_nvme4a/wandb/navier-stokes-nms-40-64-wonorm/inr/meta0.4_wf_r3_last40_T64_128sub0_2_wonorm.pt'
name='eval_meta'$grid_ratio'_wf_r3_last40_T64_128sub0_2_wonorm'
id='eval_meta'$grid_ratio'_wf_r3_S256_last40_T64_128sub0_2_wonorm'
python3 inr/inr_metagrid.py "optim.grid_ratio=$grid_ratio" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" "wandb.checkpoint_path=$checkpoint_path" #"wandb.dir=$dir" 
