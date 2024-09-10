# this script is for getting the results from sdf file, and convert them into csv
# it is used to evaluation GNINA results
import glob
import os.path
from collections import defaultdict

import pandas as pd
from rdkit import Chem

root = '/opt/home/revoli/data_worker/paper/benchmark/affinity/kinase/E8M9_20240820AE_EXPORT'
sdf_root = f'{root}/infer'
# df = pd.read_csv(f'{root}/kinase_test.csv')

data = []
for sdf_f in glob.glob(f'{root}/*.sdf'):
    try:
        supplier = Chem.SDMolSupplier(sdf_f, sanitize=True, removeHs=False)
    except:
        continue
    names = defaultdict(int)
    for mol in supplier:
        if mol:
            t = os.path.basename(sdf_f)[:4]
            name = mol.GetProp('_Name')
            vina = float(mol.GetProp('minimizedAffinity'))
            gnina = float(mol.GetProp('CNNaffinity'))
            data.append([t, name, names[name], vina, gnina])
            names[name] += 1

dock_data = pd.DataFrame(data, columns=['Target', 'Molecule ID', 'pose_rank', 'vina', 'gnina'])
# merge = df.merge(dock_data, on=['Target', 'Molecule ID'])
# Merge with pGNN
# pgnn = pd.read_csv('/opt/home/revoli/git/pf-gnn_pli/GNNp_regression/kinase_test_ensemble.csv')
# pgnn = pgnn[['Target', 'Molecule ID', 'pred_pIC50']]
# pgnn = pgnn.rename(columns={'pred_pIC50': 'pGNN'})
# merg2 = merge.merge(pgnn, on=['Target', 'Molecule ID'])
# output
dock_data.to_csv(f'{root}/kinase_test.gnina.csv', index=False)
