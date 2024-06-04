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
dir='/cluster/nvme4a/whh/dataset/ns_nms/u_Re800_T60_transient_tr256.npy'
dataset_name="navier-stokes-nms-40-64-wonorm"
same_grid=True
sub_from=2
sub_tr=0.2
sub_te=0.2
seq_inter_len=20
seq_extra_len=20
batch_size=64

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

# saved_checkpoint=True
# evaluate=True 
# checkpoint_path='/cluster/home1/whh/workspace/wandb/sst-11-22/model/dyn_grid64_V0_bs128_sst_zone_full_'$zone_id'_ck.pt'

grid_size=$1
grid_channel=3
run_name_suffix=''

#run_name="toasty-darkness-5007"  #"desert-sponge-4958" # "eager-field-4969" # splendid-yogurt-4959 # "desert-sponge-4958" # misunderstood-sunset-4916
run_name='grid'$grid_size'_woscale_bs128_wf_r3_last40_T64_128sub0_2-wonorm2'
name='dyn_grid'$grid_size'_woscale_bs128_wf_r3_last40_T64_128sub0_2-wonorm2_debug'
id='dyn_grid'$grid_size'_woscale_bs128_wf_r3_S256_last40_T64_128sub0_2-wonorm2_debug'
python3 dynamics_modeling/train_grid_debug.py "inr.grid_channel=$grid_channel" "inr.run_name_suffix=$run_name_suffix" "inr.grid_size=$grid_size" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.id=$id" #"wandb.evaluate=$evaluate" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.checkpoint_path=$checkpoint_path" 
