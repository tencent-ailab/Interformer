import torch
from rdkit import Chem

import os
from dataset import MolDataset, collate_fn, get_atom_feature
from torch.utils.data import DataLoader
from gnn import gnn
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
    parser.add_argument("--epoch", help="epoch", type=int, default=10000)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=7)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)

    parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default='./save/')
    parser.add_argument("--data_dir", help="data to numpy features directory", type=str, default='./data_numpy/')
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=4.0)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=1.0)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.0)
    parser.add_argument("--train_keys", help="train keys", type=str, default='keys/pdbbind_dude_small_train.pkl')
    parser.add_argument("--test_keys", help="test keys", type=str, default='keys/pdbbind_dude_small_test.pkl')
    args = parser.parse_args()
    print(args)
    return args


def load_sdf(sdf_f, m_id):
    supple = Chem.SDMolSupplier(sdf_f)
    for mol in supple:
        if mol is not None:
            name = mol.GetProp('_Name')
            if name == m_id:
                return mol
    return supple[0]


def make_feat(m1, m2, m1_m2, Y):
    try:
        n1 = m1.GetNumAtoms()
        adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
        H1 = get_atom_feature(m1, True)
        # prepare protein
        n2 = m2.GetNumAtoms()
        adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
        H2 = get_atom_feature(m2, False)
        # no aggregation here
        # node indice for aggregation - kept to be used later on in the model
        valid = np.zeros((n1 + n2,))
        valid[:n1] = 1
        sample = {
            'H1': H1,
            'H2': H2,
            'A1': adj1,
            'A2': adj2,
            'Y': Y,
            'V': valid,
            'key': m1_m2,
        }
        # np.savez_compressed(numpy_dir + "/" + m1_m2, features=sample)
        return sample
        # print("hi")
    except Exception as e:
        print("Exception occured =====", e)


if __name__ == '__main__':
    data_dir = '/opt/home/revoli/data_worker/paper/benchmark/affinity'
    input_csv = f'{data_dir}/mPro_project.csv'  # lsd1_project, mPro_covalent_test, mPro_project
    ####
    # Core
    # data_dir = '/opt/home/revoli/data_worker/interformer/poses'
    # input_csv = '/opt/home/revoli/data_worker/paper/benchmark/docking/timesplit/core_timetest.round0.csv'
    #
    output_dir = '/opt/home/revoli/git/pf-gnn_pli/GNNp_regression'

    args = get_args()
    args.batch_size = 32
    args.num_workers = 0
    #
    df = pd.read_csv(input_csv)
    # for core set
    # df = df[(df['pose_rank'] == 20) & (df['task'] == 'core')].reset_index(drop=True)
    #
    feats = []
    indices = []
    for i, row in tqdm(df.iterrows()):
        pdb, m_id = row['Target'], row['Molecule ID']
        label = row['pIC50']
        # sdf_f = f'{data_dir}/ligand/final/{pdb}_docked.sdf'
        sdf_f = f'{data_dir}/infer/{pdb}_docked.sdf'
        sdf_mol = load_sdf(sdf_f, m_id)
        pdb_mol = Chem.MolFromPDBFile(f'{data_dir}/pocket/{pdb}_pocket.pdb')
        feat = make_feat(sdf_mol, pdb_mol, f'{pdb}-{m_id}', Y=label)
        if feat is not None:
            indices.append(i)
            feats.append(feat)
    df = df.iloc[indices]
    # Inference
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataloader = DataLoader(feats, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    model = gnn(args).to(device)
    state = torch.load('/opt/home/revoli/git/pf-gnn_pli/GNNp_regression/save/save_100.pt')
    state = {key.replace('module.', ''): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    test_true = []
    test_pred = []
    for i_batch, sample in enumerate(test_dataloader):
        model.zero_grad()
        H1, H2, A1, A2, Y, V, keys = sample
        H1, H2, A1, A2, Y, V = H1.to(device), H2.to(device), A1.to(device), A2.to(device), \
            Y.to(device), V.to(device)

        # train neural network
        pred = model.train_model((H1, H2, A1, A2, V))

        # collect loss, true label and predicted label
        test_true.append(Y.data.cpu().numpy())
        test_pred.extend(pred.data.cpu().numpy())

    test_pred = np.array(test_pred)
    df['pred_pIC50'] = test_pred
    print(df.corr(numeric_only=True)['pIC50'])
    print('=' *  100)
    print(df.corr('spearman', numeric_only=True)['pIC50'])
    print(len(df))
    df.to_csv(f'{output_dir}/{os.path.basename(input_csv)[:-4]}_ensemble.csv', index=False)
    print(test_pred)
