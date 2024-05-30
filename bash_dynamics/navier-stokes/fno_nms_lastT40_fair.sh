#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=fno
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/fno/%x-%j.out
#SBATCH --error=slurm_run/fno/%x-%j.err

# source $MINICONDA_PATH/etc/profile.d/conda.sh

# set -x
# conda init bash
# conda activate fno

dataset_name='navier-stokes-nms'
dir='/cluster/home1/whh/workspace/Neural-Spectral-Methods/src/pde/navierstokes/u_Re800_T60_transient_tr256.npy'
sub_from=1
sub_tr=4
sub_te=4
setting=extrapolation
seq_inter_len=20
seq_extra_len=20

batch_size=64
learning_rate=0.001
epochs=10000
scheduler_step=100
scheduler_gamma=0.75
modes=12
width=32
name='fno_nms_wf_r3_last40_fair_wonorm' 

python3 baseline/fno/train.py "data.dir=$dir" "data.dataset_name=$dataset_name" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.sequence_length_optim=$sequence_length_optim" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "fno.modes=$modes" "fno.width=$width" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "optim.learning_rate=$learning_rate" "optim.scheduler_step=$scheduler_step" "optim.scheduler_gamma=$scheduler_gamma" "wandb.name=$name"
