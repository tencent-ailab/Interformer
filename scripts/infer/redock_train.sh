TRAIN_FOLDER=~/data_worker
DOCK_FOLDER=dock_results/energy_train
PYTHONPATH=interformer/ python inference.py -test_csv $TRAIN_FOLDER/train/general_PL_2020.csv \
-work_path $TRAIN_FOLDER/poses \
-ensemble checkpoints/v0.2_energy_model \
-ligand_folder ligand/final \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_FOLDER \
-reload
