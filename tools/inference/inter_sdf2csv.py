# This script is for converting sdf into the input format of interformer(csv).
import os.path
from collections import defaultdict

from rdkit import Chem
import sys
import pandas as pd


def merge_label(df1, df2):
    df2 = df2[df2['pose_rank'] == 0] if 'pose_rank' in df2 else df2
    df2 = df2.drop(columns=['pose_rank'])
    df = df1.merge(df2, on=['Target', 'Molecule ID'])
    print(df.corr(numeric_only=True))
    return df


def make_csv_by(sdf_f, target, isuff=True, label_df=None):
    data = []
    suppl = Chem.SDMolSupplier(sdf_f)
    for i, mol in enumerate(suppl):
        if mol is not None:
            m_id = mol.GetProp('_Name')
            if isuff:
                data.append([target, 0, i, m_id])
            else:
                data.append([target, i, 0, m_id])

    df = pd.DataFrame(data, columns=['Target', 'pose_rank', 'uff_pose_rank', 'Molecule ID'])
    # merge label
    if label_df is not None:
        df = merge_label(df, label_df)
    base = os.path.basename(sdf_f)[:-4]
    df.to_csv(os.path.dirname(sdf_f) + f'/{base}_infer.csv', index=False)


if __name__ == '__main__':
    # this script convert prepared sdf into csv, for the inference of interformer
    print("PDB_ID=sdf_f[:4]")
    print('[sdf_f] [IsUff, 0|1] [pIC50_csv]')
    sdf_f = sys.argv[1]
    pdb = os.path.basename(sdf_f)[:4]
    IsUff = bool(int(sys.argv[2]))
    if len(sys.argv) > 3:
        label_df = pd.read_csv(sys.argv[3])
    else:
        label_df = None
    # Making Csv file
    make_csv_by(sdf_f, pdb, IsUff, label_df)
