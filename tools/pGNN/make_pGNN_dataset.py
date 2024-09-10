import glob
import os.path
import pickle

import pandas as pd
from rdkit import Chem

if __name__ == '__main__':
    root = '/opt/home/revoli/data_worker/interformer/poses'
    # Mol data
    ligand = list(glob.glob(f'{root}/ligand/*.sdf'))
    output_root = '/opt/home/revoli/git/pf-gnn_pli/GNNp_regression/data_mol_pkl'
    for sdf_f in ligand:
      pdb = os.path.basename(sdf_f)[:4]
      pocket_mol = Chem.MolFromPDBFile(f'{root}/pocket/{pdb}_pocket.pdb')
      sdf_mol = Chem.SDMolSupplier(sdf_f)[0]
      pickle.dump([sdf_mol, pocket_mol], open(f'{output_root}/{pdb}', 'wb'))
    ####
    # pIC50 data
    output_root = '/opt/home/revoli/git/pf-gnn_pli/GNNp_regression/keys'
    df = pd.read_csv('/opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv')
    df = df[['Target', 'pIC50']]
    train_pdb = [x.strip() for x in open(
        '/opt/home/revoli/data_worker/interformer/train/diffdock_splits/timesplit_no_lig_overlap_train').readlines()]
    test_pdb = [x.strip() for x in
                open('/opt/home/revoli/data_worker/interformer/train/diffdock_splits/timesplit_test').readlines()]
    coreset = [x.strip() for x in
               open('/opt/home/revoli/data_worker/interformer/train/diffdock_splits/coresetlist').readlines()]
    train_pdb = list(set(train_pdb) - set(coreset))
    #
    tmp = df[df['Target'].isin(train_pdb)]
    train = tmp.set_index('Target')['pIC50'].to_dict()
    tmp = df[df['Target'].isin(test_pdb)]
    test = tmp.set_index('Target')['pIC50'].to_dict()
    all = df.set_index('Target')['pIC50'].to_dict()
    pickle.dump(all, open(f'{output_root}/PDBBind_pIC50_sample_labels.pkl', 'wb'))
    pickle.dump(train, open(f'{output_root}/PDBBind_pIC50_sample_train_keys.pkl', 'wb'))
    pickle.dump(test, open(f'{output_root}/PDBBind_pIC50_sample_test_keys.pkl', 'wb'))
    #
    print('done')
