#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=deeponet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/deeponet/%x-%j.out
#SBATCH --error=slurm_run/deeponet/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

dataset_name='navier-stokes-dino'
same_grid=False
sub_from=4
sub_tr=0.2
sub_te=0.2
setting=extrapolation
seq_inter_len=20
seq_extra_len=20

batch_size=40
learning_rate=0.00001
epochs=10000

model_type="mlp"
trunk_depth=4
branch_depth=4
width=100

<<<<<<< HEAD
python3 baseline/deeponet/train.py "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.setting=$setting" "deeponet.model_type=$model_type" "deeponet.branch_depth=$branch_depth" "deeponet.trunk_depth=$trunk_depth" "deeponet.width=$width" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "optim.learning_rate=$learning_rate" "data.dataset_name=$dataset_name" 
=======
python3 deeponet/coral/deeponet_2d_time.py "data.same_grid=$same_grid" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.sequence_length_optim=$sequence_length_optim" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "deeponet.model_type=$model_type" "deeponet.branch_depth=$branch_depth" "deeponet.trunk_depth=$trunk_depth" "deeponet.width=$width" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "optim.learning_rate=$learning_rate" 
>>>>>>> d17b5b64186022ce4c2f28b7159abea92bbb920d
