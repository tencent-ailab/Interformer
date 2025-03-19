###
# Energy
PYTHONPATH=interformer/ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
-work_path /opt/home/revoli/data_worker/interformer/poses \
-ligand ligand/rcsb \
-seed 1111 \
-filter_type normal \
-native_sampler 0 \
-Code Energy \
-batch_size 24 \
-gpus 4 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 11000 \
--peak_lr 0.0012 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5 \
--energy_mode True
#####
# Affinity&PoseScore
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020_round0_full.csv \
-work_path /opt/home/revoli/data_worker/interformer/poses \
-seed 1111 \
-filter_type full \
-native_sampler 0 \
-Code affinity \
-batch_size 24 \
-gpus 4 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 10000 \
--peak_lr 0.0008 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5 \
--pose_sel_mode True
#####
# Affinity Normal
PYTHONPATH=interformer/ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
-work_path /opt/home/revoli/data_worker/interformer/poses \
-ligand ligand/rcsb \
-seed 1111 \
-filter_type normal \
-native_sampler 1 \
-Code affinity_normal \
-batch_size 24 \
-gpus 4 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 10000 \
--peak_lr 0.0008 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5