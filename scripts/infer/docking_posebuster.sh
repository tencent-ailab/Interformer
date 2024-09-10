DOCK_FOLDER=dock_results/energy_posebuster
WORK_PATH=benchmarks/docking/posebuster
mkdir -p DOCK_FOLDER
######
PYTHONPATH=interformer/ python inference.py -test_csv $WORK_PATH/posebuster_infer.csv  \
-work_path $WORK_PATH \
-ensemble checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_FOLDER \
-reload

# Using refer conformation
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all --output_folder ref find  # docking with ref ligands
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all --output_folder ref stat

# Using start conformation
# OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all --uff_folder uff find
# python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all stat


# testing
#PDB=6Z0R
#bust ${PDB}_docked.sdf -l ../ligand/${PDB}_docked.sdf -p ../pocket/${PDB}_pocket.pdb  --outfmt short
# ls *.sdf | cut -c -4 | xargs -I {}  bust {}_docked.sdf -l ../ligand/{}_docked.sdf -p ../pocket/{}_pocket.pdb  --outfmt long --top-n 1  # quick test
