import pandas as pd
import os
from rdkit import Chem

df = pd.read_csv('/opt/home/revoli/eva/Interformer/result/posebuster_infer.round0_ensemble.csv')
dock_sdf_root = '/opt/home/revoli/eva/Interformer/dock_results/energy_posebuster/ligand_reconstructing'
output_root = '/opt/home/revoli/eva/Interformer/dock_results/energy_posebuster/posescore'
os.makedirs(output_root, exist_ok=True)
df = df.sort_values('pred_pose', ascending=False)
df = df.groupby('Target').head(1)
for index, row in df.iterrows():
    t, rank = row['Target'], int(row['pose_rank'])
    mol = Chem.SDMolSupplier(f'{dock_sdf_root}/{t}_docked.sdf')[rank]

    w = Chem.SDWriter(f'{output_root}/{t}_docked.sdf')
    w.write(mol)
    w.close()
print('done')
