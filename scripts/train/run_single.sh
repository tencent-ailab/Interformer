#################
# --pose_sel True \
# CUDA_VISIBLE_DEVICES=4,5,6,7
# CUDA_VISIBLE_DEVICES=0,1,2,3
# -filter_type hinge \
########
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020_round0.csv \
#-work_path /opt/home/revoli/data_worker/interformer/poses \
#-dataset sbdd \
#-filter_type hinge \
#-native_sampler 0 \
#-Code Round0_affinity \
#--warmup_updates \
#10000 \
#--peak_lr \
#0.0005 \
#-model \
#Interformer \
#-batch_size \
#6 \
#-gpus \
#4 \
#-method \
#Gnina2 \
#--n_layers \
#6 \
#--hidden_dim \
#128 \
#--num_heads \
#8 \
#--dropout_rate \
#0.1 \
#--attention_dropout_rate \
#0.1 \
#--weight_decay \
#1e-4 \
#-patience 20 \
#-early_stop_metric val_loss \
#-early_stop_mode min \
#--pose_sel True
#####
# Affinity
# general_PL_2020_round0_full
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020_round0_full.csv \
#-work_path /opt/home/revoli/data_worker/interformer/poses \
#-dataset sbdd \
#-filter_type full \
#-per_target_sampler 40000 \
#-Code affinity_normal \
#--warmup_updates \
#10000 \
#--peak_lr \
#0.0005 \
#-model \
#Interformer \
#-batch_size \
#8 \
#-gpus \
#4 \
#-method \
#Gnina2 \
#--n_layers \
#6 \
#--hidden_dim \
#128 \
#--num_heads \
#8 \
#--dropout_rate \
#0.1 \
#--attention_dropout_rate \
#0.1 \
#--weight_decay \
#1e-4 \
#-patience 20 \
#-early_stop_metric val_loss \
#-early_stop_mode min \
#--pose_sel True \
#--energy_mode False
#####
# Normal
# -native_sampler 1 \
# 4,5,6,7
# 0,1,2,3
# MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0
# CUDA_LAUNCH_BLOCKING=0
# -per_target_sampler 20000 \
# -split_folder gnina_fold0 \
#CUDA_VISIBLE_DEVICES=0,1,2,4 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
#-work_path /opt/home/revoli/data_worker/interformer/poses \
#-dataset sbdd \
#-seed 1111 \
#-filter_type normal \
#-per_target_sampler 0 \
#-native_sampler 0 \
#-Code affinity_normal \
#--warmup_updates \
#10000 \
#--peak_lr \
#0.001 \
#-model \
#Interformer \
#-batch_size \
#6 \
#-gpus \
#4 \
#-method \
#Gnina2 \
#--n_layers \
#6 \
#--hidden_dim \
#128 \
#--num_heads \
#8 \
#--dropout_rate \
#0.1 \
#--attention_dropout_rate \
#0.1 \
#--weight_decay \
#1e-4 \
#-patience 60 \
#-early_stop_metric val_loss \
#-early_stop_mode min
####
# Normal
PYTHONPATH=interformer/ CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
-work_path /opt/home/revoli/data_worker/interformer/poses \
-dataset sbdd \
-seed 1111 \
-filter_type normal \
-per_target_sampler 0 \
-native_sampler 1 \
-Code Normal \
--warmup_updates \
10000 \
--peak_lr \
0.0006 \
-model \
Interformer \
-batch_size \
10 \
-gpus \
4 \
-method \
Gnina2 \
--n_layers \
6 \
--hidden_dim \
128 \
--num_heads \
8 \
--dropout_rate \
0.1 \
--attention_dropout_rate \
0.1 \
--weight_decay \
1e-5 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre