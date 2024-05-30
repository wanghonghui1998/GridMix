
dir='/cluster/home1/whh/workspace/Neural-Spectral-Methods/src/pde/navierstokes/u_Re800_T60_transient_tr256.npy'
dataset_name='navier-stokes-nms'
model_type='siren'
same_grid=True
sub_from=1
sub_tr=4
sub_te=4
seq_inter_len=20
seq_extra_len=20
batch_size=128
lr_inr=0.000005
lr_edg=0.001
epochs=2000
latent_dim=128
depth=4
hidden_dim=128
saved_checkpoint=False
temperature=$1
w_cl=$2
name='wf_r3_firstT40_edg_1e-3_wnorm_cl'$w_cl'_tau'$temperature
id='wf_r3_S256_firstT40_edg_1e-3_wnorm_cl'$w_cl'_tau'$temperature'_2k'
use_cl=True

launcher='slurm' 
port=29052

python3 inr/inr_edg_ddp.py "optim.use_cl=$use_cl" "optim.w_cl=$w_cl" "optim.temperature=$temperature" "optim.lr_edg=$lr_edg" "data.dir=$dir" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "inr.model_type=$model_type" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" "wandb.name=$name" "wandb.id=$id" #"wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
# "ddp.launcher=$launcher" "ddp.port=$port" 