#!/bin/bash
#SBATCH --partition=hard
#SBATCH --constraint "GPUM48G"
#SBATCH --job-name=dino
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/dino/%x-%j.out
#SBATCH --error=slurm_run/dino/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

# data
dataset_name='shallow-water-dino'
sub_from=2
sub_tr=0.05
sub_te=0.05
seq_inter_len=20 
seq_extra_len=20
same_grid=False

# optim
epochs=10000
lr=0.001
minibatch_size=4
minibatch_val_size=4

# inr
state_dim=2
code_dim=200
hidden_c_enc=256
n_layers=6
coord_dim=3

# forecasster 
hidden_c=800
teacher_forcing_init=0.99
teacher_forcing_decay=0.99
teacher_forcing_update=5

python3 baseline/dino/train.py "data.dataset_name=$dataset_name" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.epochs=$epochs" "optim.minibatch_size=$minibatch_size" "optim.minibatch_val_size=$minibatch_val_size" "optim.lr=$lr" "inr.state_dim=$state_dim" "inr.code_dim=$code_dim" "inr.hidden_c_enc=$hidden_c_enc" "inr.n_layers=$n_layers" "inr.coord_dim=$coord_dim" "forecaster.hidden_c=$hidden_c" "forecaster.teacher_forcing_update=$teacher_forcing_update"