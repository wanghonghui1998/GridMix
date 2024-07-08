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
dir='/cluster/home1/whh/new_repo/DINo/results'
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


inr_batch_size=$4
w0=$5
seed=$6
meta_lr_code=$7

saved_checkpoint=False
#run_name="toasty-darkness-5007"  #"desert-sponge-4958" # "eager-field-4969" # splendid-yogurt-4959 # "desert-sponge-4958" # misunderstood-sunset-4916
run_name='2coral_mlc'$meta_lr_code'_bs'$inr_batch_size'_w0'$w0'_seed'$seed'_ns_dino_v0_r3_256sub'$1'_'$2'_'$3
name='full_dyn_coral_mlc'$meta_lr_code'_bs'$inr_batch_size'_w0'$w0'_seed'$seed'_ns_dino_v0_r3_256sub'$1'_'$2'_'$3
id='full_dyn_coral_mlc'$meta_lr_code'_bs'$inr_batch_size'_w0'$w0'_seed'$seed'_ns_dino_v0_r3_256sub'$1'_'$2'_'$3
python3 dynamics_modeling/train.py "optim.gamma_step=$gamma_step" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.id=$id" "wandb.saved_checkpoint=$saved_checkpoint" # "wandb.checkpoint_path=$checkpoint_path" "wandb.evaluate=$evaluate"
