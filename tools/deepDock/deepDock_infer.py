import glob
import os.path
import joblib
from rdkit import Chem
import deepdock
from deepdock.models import *
from deepdock.DockingFunction import dock_compound, get_random_conformation

import numpy as np
import torch

# set the random seeds for reproducibility
np.random.seed(123)
torch.cuda.manual_seed_all(123)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ligand_model = LigandNet(28, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(4, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10, dist_threhold=7.).to(
    device)

checkpoint = torch.load('./Trained_models/DeepDock_pdbbindv2019_13K_minTestLoss.chk', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

def main(row):
    pdb, sdf_f, target_ply = row
    # real_mol = Chem.MolFromMol2File('1z6e_ligand.mol2',sanitize=False, cleanupSubstructures=False)
    real_mol = Chem.SDMolSupplier(sdf_f, sanitize=True)[0]
    if real_mol is None:
        print(f"error:{pdb}")
        return
    opt_mol, init_mol, result = dock_compound(real_mol, target_ply, model, dist_threshold=3., popsize=150, seed=123,
                                              device=device)
    print(result)
    print(opt_mol)
    # write it out
    w = Chem.SDWriter(f'{output_folder}/{pdb}_docked.sdf')
    w.write(opt_mol)
    w.close()


print("loaded.")
if __name__ == '__main__':
    root = '/data/timesplit/pocket'
    output_folder = '/data/output'
    os.makedirs(output_folder, exist_ok=True)
    data = []
    for sdf_f in glob.glob(f'{root}/*.sdf'):
        pdb = os.path.basename(sdf_f)[:4]
        print(pdb)
        target_ply = f'{root}/{pdb}_pocket.ply'
        if not os.path.exists(target_ply):
            continue
        if os.path.exists(f'{output_folder}/{pdb}_docked.sdf'):
            continue
        data.append([pdb, sdf_f, target_ply])

    joblib.Parallel(n_jobs=10)(joblib.delayed(main)(x) for x in data)
