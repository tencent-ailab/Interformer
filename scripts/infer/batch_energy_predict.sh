PYTHONPATH=interformer/ python inference.py -test_csv /opt/home/revoli/data_worker/project/cfDNA/cfDNA.csv  \
-work_path /opt/home/revoli/data_worker/project/cfDNA \
-ensemble checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder energy_cfDNA \
-uff_as_ligand \
-reload

OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd ./energy_cfDNA --find_all --uff_folder uff find