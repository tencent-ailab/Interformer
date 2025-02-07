This is a virtual screening demonstration where numerous ligands are docked into a rigid protein pocket, and their affinity values are scored. For this example, we are using a dataset located in the `examples/` folder.
```
example/
├── ligand
│   └── 2qbr_docked.sdf (Ensure that it is a bound ligand conformation inside the binding pocket, as it is the best choice. This sdf file can be obtained from RCSB website.)
├── pocket
│   └── 2qbr_pocket.pdb (A prepared protein structure, it can be the entire protein or pocket structure)
└── uff
    └── 2qbr_uff.sdf (Prepare a force filed minimized SDF file containing all the ligands you wish to dock.)
```
Below is the script for virtual screening on one protein with many ligands.
```
MAIN=~/Interformer
# 1. Create query csv
python $MAIN/tools/inference/inter_sdf2csv.py example/uff/2qbr_uff.sdf 1
# 2. Predict energy files
PYTHONPATH=$MAIN/interformer python $MAIN/inference.py -test_csv example/uff/2qbr_uff_infer.csv \
-work_path example/ \
-ensemble $MAIN/checkpoints/v0.2_energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder energy_output \
-uff_as_ligand \
-debug \
-reload
# 3. Docking
# [VS mode] need to refresh the uff ligand file
cp example/uff/2qbr_uff.sdf energy_output/uff/
OMP_NUM_THREADS="64,64" python $MAIN/docking/reconstruct_ligands.py -y --cwd ./energy_output --find_all --uff_folder uff find
# 4. Scoring
cp -r energy_output/ligand_reconstructing/ example/infer/
python $MAIN/tools/inference/inter_sdf2csv.py example/infer/2qbr_docked.sdf 0

PYTHONPATH=$MAIN/interformer/ python $MAIN/inference.py -test_csv example/infer/2qbr_docked_infer.csv  \
-work_path example/ \
-ligand_folder infer/ \
-ensemble $MAIN/checkpoints/v0.2_affinity_model/model* \
-use_ff_ligands '' \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True

# 5. Review result
cat result/2qbr_docked_infer_ensemble.csv
```