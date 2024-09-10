import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit import Chem
import joblib
from rdkit import DataStructs


def morgan_fp(sdf_f):
    mol = Chem.SDMolSupplier(sdf_f)[0]
    if mol is not None:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp = fpgen.GetFingerprint(mol)
        return fp
    return None


def grep_max_simiarity(query_smiles, train_fps):
    query_mol = Chem.MolFromSmiles(query_smiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    query_fp = fpgen.GetFingerprint(query_mol)
    max_sim = 0.
    for fps in train_fps:
        if fps is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, fps)
            max_sim = max(max_sim, sim)
    return max_sim


if __name__ == '__main__':
    # compare simiarity of ligand to training set
    pre = 0
    test_root = '/opt/home/revoli/data_worker/paper/benchmark/affinity'
    output_dir = '/opt/home/revoli/data_worker/paper/benchmark/docking/similarity'
    tmp_pkl_f = '/tmp/train_fps.pkl'
    os.makedirs(output_dir, exist_ok=True)
    input_csv = ['mPro_covalent_test.csv', 'mPro_project.csv', 'kinase_test.csv', 'lsd1_project.csv']
    # create train morganFP first
    if pre:
        train_csv = pd.read_csv('/opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv')
        train_ligand_root = '/opt/home/revoli/data_worker/interformer/poses/ligand/final'
        pdbs = train_csv['Target'].unique().tolist()
        train_fps = []
        for pdb in pdbs:
            train_fps.append(morgan_fp(f'{train_ligand_root}/{pdb}_docked.sdf'))
        pickle.dump(train_fps, open(tmp_pkl_f, 'wb'))
    # Query
    train_fps = pickle.load(open(tmp_pkl_f, 'rb'))
    for csv_f in input_csv:
        csv = f'{test_root}/{csv_f}'
        query_df = pd.read_csv(csv)
        query_df = query_df.groupby('Smiles').head(1)  # some csv may contains duplicate smiles,
        # we only care about ligand
        # query
        sims = []
        for i, row in query_df.iterrows():
            query_smiles = row['Smiles']
            sim = grep_max_simiarity(query_smiles, train_fps)
            sims.append(sim)
        query_df['simiarity'] = sims
        ######
        query_df.to_csv(f'{output_dir}/{csv_f}', index=False)
        print(csv)
        print(f'n:{len(query_df)}, max:{query_df['simiarity'].max()}, median:{query_df['simiarity'].median()}')
        print('=' * 100)

    print('done')
