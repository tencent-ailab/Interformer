# Installation

```
apt-get install -y build-essential
conda env update --file envfile_conda_base.yml
conda env update --file envfile_conda_env_for_obabel.yml
pip install .
```

# Demo

## reconstruct 1 ligand

```
export OMP_NUM_THREADS="1,64"  # you may setup the number of threads used to docking.
cd ./examples/

python ../reconstruct_1_ligand.py  \
    --sdf_ligand uff/1bcu_uff.sdf  \ # input ligand, its conformation and position will be randomized at the beginging of sampling.
    --sdf_ref ligand/1bcu_docked.sdf  \ # reference ligand used for intercepting of pocket
    --pdb_complex complex/1bcu_complex.pdb  \  # aligned complex with respect to the energy funtions predicted by interformer  
    --pkl_normalscore gaussian_predict/1bcu_G.pkl  \ # interformer's predicted energy funtions
    --sdf_output ./1bcu_demo.sdf  # the output file for the docking pose.

python ../reconstruct_1_ligand.py  \
    --sdf_ligand ligand/2qbq_docked.sdf  \  
    --sdf_ref ligand/2qbq_docked.sdf  \
    --pdb_complex complex/2qbq_complex.pdb  \
    --pkl_normalscore gaussian_predict/2qbq_G.pkl  \
    --sdf_output ligand_reconstructing/2qbq_docked.sdf  \
    --csv_stat ligand_reconstructing/2qbq_docked.sdf_stat.csv  \  # docking summary csv file.
    --pdb_id 2qbq \  # [optional] for record in summary file.
    --weight_intra 30.0  # [weight of the intra-term] \
    --weight_collision_inter 40.0
```

## reconstruct all ligands and calculate Top1

### reproduce paper result (top1 = 0.56)

```
cd ./examples/

python ../reconstruct_ligands.py --cwd ./ --find_all --yes --weight_intra 0.0 find
python ../reconstruct_ligands.py --cwd ./ --find_all stat
```

### better performance (top1 = 0.70)

```
cd ./examples/

python ../reconstruct_ligands.py --cwd ./ --find_all --yes --weight_intra 30.0 find  # NEW features!! 20240320, a weight to intra-term(steric clash penalty term only), in order to prevent the ligand itself from having steric clash within itself.
python ../reconstruct_ligands.py --cwd ./ --find_all stat
```

### Copyright

pyvina program contains some of the code fork from idock https://github.com/stcmz/jdock/tree/v2.2.1.
