WORK_PATH=benchmarks/affinity

PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/input_csv/lsd1_project.csv ${WORK_PATH}/input_csv/mPro_covalent_test.csv ${WORK_PATH}/input_csv/mPro_project.csv \
-work_path ${WORK_PATH} \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-use_ff_ligands '' \
-use_mid \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True

# below is kinase affinity prediction, it may take a little bit long time to predict.
PYTHONPATH=interformer/ python inference.py -test_csv ${WORK_PATH}/input_csv/kinase_test.csv \
-work_path ${WORK_PATH} \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-use_ff_ligands '' \
-use_mid \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True
