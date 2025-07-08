PDB=2qbr
WORK_PATH=vs_example
DOCKING_PATH=energy_VS

# 1. Create query csv
python tools/inference/inter_sdf2csv.py ${WORK_PATH}/uff/${PDB}_uff.sdf 1
# 2. Predict energy files
PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/uff/${PDB}_uff_infer.csv \
-work_path ${WORK_PATH}/ \
-ensemble checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder ${DOCKING_PATH} \
-uff_as_ligand \
-debug \
-reload
# 3. Docking
# [VS mode] need to refresh the uff ligand file
cp ${WORK_PATH}/uff/${PDB}_uff.sdf ${DOCKING_PATH}/uff/
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd ${DOCKING_PATH} --find_all --uff_folder uff find
# 4. Scoring
cp -r ${DOCKING_PATH}/ligand_reconstructing/ ${WORK_PATH}/infer/
python tools/inference/inter_sdf2csv.py ${WORK_PATH}/infer/${PDB}_docked.sdf 0

PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/infer/${PDB}_docked_infer.csv  \
-work_path ${WORK_PATH}/ \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-use_ff_ligands '' \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True

# 5. Review result
cat result/${PDB}_docked_infer_ensemble.csv