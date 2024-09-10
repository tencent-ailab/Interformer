DOCK_OUTPUT_DIR=dock_results/energy_timetest
WORK_PATH=benchmarks/docking/timesplit
mkdir -p $DOCK_OUTPUT_DIR
####
# Energy Predictions
PYTHONPATH=interformer/ python inference.py -test_csv $WORK_PATH/core_timetest.csv \
-work_path $WORK_PATH \
-ensemble checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_OUTPUT_DIR \
-reload
####
# Docking
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_OUTPUT_DIR --find_all find
python docking/reconstruct_ligands.py -y --cwd $DOCK_OUTPUT_DIR --find_all stat

python docking/merge_summary_input.py $DOCK_OUTPUT_DIR/ligand_reconstructing/stat_concated.csv $WORK_PATH/core_timetest.csv  # gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
mv $DOCK_OUTPUT_DIR/ligand_reconstructing $WORK_PATH/infer
####
# PoseScore-Affinity
PYTHONPATH=interformer/ python inference.py -test_csv $WORK_PATH/core_timetest.round0.csv \
-work_path $WORK_PATH \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True
###
# Docking AC Evaluation
python eda/docking/eval_docking_ac.py result/core_timetest.round0_ensemble.csv
