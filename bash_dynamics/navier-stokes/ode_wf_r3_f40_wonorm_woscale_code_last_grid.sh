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

grid_size=32
grid_channel=1
run_name_suffix=''
grid_ratio=0.4 
lr_code=1e-2
inner_steps=3 
modes=12
normalize_per_ele=False
normalize_per_channel=True
model_type='code_grid'
evaluate=False
#run_name="toasty-darkness-5007"  #"desert-sponge-4958" # "eager-field-4969" # splendid-yogurt-4959 # "desert-sponge-4958" # misunderstood-sunset-4916
run_name='2metacode+lastgrid'$grid_size'_woscale_wf_r3_first40_T64-wonorm_gridratio'$grid_ratio'_step'$inner_steps'_bs128_lr'$lr_code
name='dyn3_npc_metacode+lastgrid'$grid_size'_woscale_wf_r3_first40_T64-wonorm_gridratio'$grid_ratio'_step'$inner_steps'_bs128_lr'$lr_code
id='dyn3_npc_metacode+lastgrid'$grid_size'_woscale_wf_r3_first40_T64-wonorm_gridratio'$grid_ratio'_step'$inner_steps'_bs128_lr'$lr_code
python3 dynamics_modeling/train_grid_code.py "dynamics.model_type=$model_type" "dynamics.normalize_per_channel=$normalize_per_channel" "dynamics.normalize_per_ele=$normalize_per_ele" "dynamics.modes=$modes" "inr.grid_channel=$grid_channel" "inr.run_name_suffix=$run_name_suffix" "inr.grid_size=$grid_size" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_name=$run_name" "wandb.name=$name" "wandb.id=$id" "wandb.evaluate=$evaluate"
