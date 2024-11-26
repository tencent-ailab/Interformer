Interformer
--------------------
Interformer is a protein-ligand complex structure prediction neural network that can predict interaction-aware energy functions for each pair of protein-ligand atoms. Such energy functions can be used in traditional protein-ligand docking sampling methods (Monte Carlo) to generate high-quality and reasonable binding poses.

The second application of the model involves using a module for pose-sensitive affinity and pose score based on contrastive learning. 
This  module assigns confidence scores to the generated docking poses and predicts corresponding affinity values. Notably, the lower the quality of the docking pose, the more probably predicting the lower affinity value (pIC50).

Below are the instructions for setting up and using the proposed method.

## Table of Contents

- [Installation](#set-up)
- [Preparation](#prepare)
- [Inference of Docking Pipeline](#docking)
- [Inference of PoseScore and Affinity Pipeline](#affinity)
- [Data Analysis](#Data-Analysis)
- [Training](#training)
- [Conclusion](#conclusion)

<a id="set-up"></a>

### Installation

Create Interformer environment using conda

```
cd Interformer
conda env create --file environment.yml
conda activate interformer

# Install PLIP without wrong openbabel dependency
pip install --no-deps plip

# Compile Docking sampling program from source code
cd docking && python setup.py install && cd ../
```


### Inference of Docking Pipeline and PoseScore/Affinity Pipeline
<a id="prepare"></a>
Here we are using the data from examples/ as demo.

The preparation of ligand and protein data involves several steps. First, hydrogen atoms are added to the ligand, and its protonation states are determined using [OpenBabel](https://github.com/openbabel/openbabel). Following this, the initial ligand conformation is generated using the UFF force field with [RDKit](https://github.com/rdkit/rdkit).

For the protein, preprocessing is carried out using the [Reduce](https://github.com/rlabduke/reduce) program, which includes the addition of hydrogen atoms along with their protonation states. The protein is then segmented into a pocket based on a 10Å radius around a reference ligand.

It should be noted that these procedures can be substituted with any other computational drug design (CADD) software. The prediction quality of the model is directly proportional to the quality of the preprocessing steps. Therefore, any improvements in this stage will likely enhance the model's predictive capabilities.

```
# Ligand
obabel examples/raw/2qbr_ligand.sdf -p 7.4 -O examples/ligand/2qbr_docked.sdf  # protonation and adding hydrogen atoms
python tools/rdkit_ETKDG_3d_gen.py examples/ligand/ examples/uff  # create inital ligand conformation using UFF

# Protein
mkdir -p examples/raw/pocket && reduce examples/raw/2qbr.pdb > examples/raw/pocket/2qbr_reduce.pdb  # prepare whole protein
python tools/extract_pocket_by_ligand.py examples/raw/pocket/ examples/ligand/ 1 && mv examples/raw/pocket/output/2qbr_pocket.pdb examples/pocket  # extract pocket 
```
1. Data should be saved as the following file structure. (We have provided the prepared data for demo, you may skip the above section)
```
examples/
├── demo_dock.csv  # the query csv for interformer prediction
├── ligand  # [reference ligand] foler contains the reference ligand conformation from PDB
├── pocket # [binding pocket site] foler contains the target protein PDB structure [can be the whole protein pdb, program will automatically cut out pocket (please remove reference ligand first)]
├── uff # [initial ligand conformation] foler contains a single ligand conformation minimized by field foce
```
<a id="docking"></a>
2. Predicting energy functions file. Download checkpoints from [zenodo](https://zenodo.org/doi/10.5281/zenodo.10828798).

```
DOCK_FOLDER=energy_output

PYTHONPATH=interformer/ python inference.py -test_csv examples/demo_dock.csv \
-work_path examples/ \
-ensemble checkpoints/v0.2_energy_model \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_FOLDER \
-reload \
-debug
```

It will produce an energy_output folder.

```
energy_output/
├── complex  # pocket intercepted through reference ligand
├── gaussian_predict  # predicted energy functions file
├── ligand  # copy from work_path ligand folder [used for locate an autobox (20Ax20Ax20A sampling space) from reference ligand]
└── uff  # copy from work_path uff folder
```

3. Giving the energy files produce docking poses via MonteCarlo sampling.

```
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all find
# --uff_folder uff # if you plan to use uff ligand as inital ligand conformation, you may add this argument.
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all stat

python docking/merge_summary_input.py $DOCK_FOLDER/ligand_reconstructing/stat_concated.csv examples/demo_dock.csv  # gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
cp -r $DOCK_FOLDER/ligand_reconstructing/ examples/infer  # move docked sdf to workpath
```
<a id="affinity"></a>
4. Giving the docking poses, predicting its PoseScore and Affinity values.

```
PYTHONPATH=interformer/ python inference.py -test_csv examples/demo_dock.round0.csv \
-work_path examples/ \
-ligand_folder infer/ \
-ensemble checkpoints/v0.2_affinity_model/model* \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* \
--pose_sel True

cat result/demo_dock.round0_ensemble.csv # you can review your final result
# energy=the predicted total sum of energy for each pair of protein-ligand atoms, pred_pIC50=the predicted affinity value, pred_pose=the predicted PoseScore(a confident score for the docking pose)
```

### FAQ

1. Reproduce the time-split, core-set, and PoseBusters docking pose from the reported paper. The parepared data in the Benchmark.zip on [zenodo](https://zenodo.org/doi/10.5281/zenodo.10828798). 

```
bash scripts/infer/docking_timesplit.sh
bash scripts/infer/docking_posebuster.sh
```
2. Reproduce the affinity benchmark results from the reported paper

```
bash scripts/infer/affinity.sh
```

<a id="Data-Analysis"></a>

### Data Analysis

please review the readme file in eda/ folder.

<a id="training"></a>

### Training

```
# Please review these scripts for further deatils

bash scripts/train/preprocess.sh # used for preprocess training data
bash scripts/train/run_single.sh # train on one single machine
```

<a id="conclusion"></a>

### Conclusion

We anticipate the Interformer will be a powerful tool in practical drug development. In future work, we plan to enhance the model by incorporating more critical specific interactions and expanding its application to a broader range of biomolecular systems, including protein-nucleic acid and protein-protein interactions.


### Citation
```
@article{Lai2024,
  title = {Interformer: an interaction-aware model for protein-ligand docking and affinity prediction},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-024-54440-6},
  DOI = {10.1038/s41467-024-54440-6},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Lai,  Houtim and Wang,  Longyue and Qian,  Ruiyuan and Huang,  Junhong and Zhou,  Peng and Ye,  Geyan and Wu,  Fandi and Wu,  Fang and Zeng,  Xiangxiang and Liu,  Wei},
  year = {2024},
  month = nov 
}
```
