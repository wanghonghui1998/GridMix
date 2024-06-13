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
dataset_name="navier-stokes-nms-f40-64-wonorm"
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
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

grid_size=32
grid_channel=3
run_name_suffix=''
normalize_per_ele=False
normalize_per_channel=True

saved_checkpoint=True
evaluate=True 
checkpoint_path='/cluster/home3/whh/workspace/pinn/exp_nvme4a/wandb/navier-stokes-nms-f40-64-wonorm/model/dyn_grid32_bs128_wf_r3_first40_T64-wonorm_V0_woscale_full_ck.pt'

#run_name="toasty-darkness-5007"  #"desert-sponge-4958" # "eager-field-4969" # splendid-yogurt-4959 # "desert-sponge-4958" # misunderstood-sunset-4916
run_name='grid32_bs128_wf_r3_first40_T64-wonorm_V0_woscale'
name='eval2_npc_dyn_grid32_bs128_wf_r3_first40_T64-wonorm_V0_woscale_full'
id='eval2_npc_dyn_grid32_bs128_wf_r3_S256_first40_T64-wonorm_V0_woscale_full'
python3 dynamics_modeling/train_grid.py "dynamics.normalize_per_channel=$normalize_per_channel" "dynamics.normalize_per_ele=$normalize_per_ele" "inr.grid_channel=$grid_channel" "inr.run_name_suffix=$run_name_suffix" "inr.grid_size=$grid_size" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.id=$id" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.checkpoint_path=$checkpoint_path" "wandb.evaluate=$evaluate"
