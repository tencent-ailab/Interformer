import glob
import random

import numpy as np
import math
from rdkit import Chem
import os
import pandas as pd
from geometry_utils import get_dihedral_vonMises, apply_changes, get_torsions, rigid_transform_Kabsch_3D
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign


def random_apply_torsion_(mol_f):
    mol = Chem.SDMolSupplier(mol_f)[0]
    mol = Chem.RemoveHs(mol)
    # obtain torsion angles
    rotable_bonds = get_torsions([mol])
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = random.random() * 100.
    optimized_mol = apply_changes(mol, new_dihedrals, rotable_bonds)
    return optimized_mol


def align(ref_f, query_f):
    # this align, have to makesure query is subset of ref, just different torsion angles meaning (query's torsion
    # angles are random from ref mol)
    ref_mol = Chem.SDMolSupplier(ref_f)[0]
    ref_mol = Chem.RemoveHs(ref_mol)
    if isinstance(query_f, str):
        query_mol = Chem.SDMolSupplier(query_f)[0]
        query_mol = Chem.RemoveAllHs(query_mol)
    else:
        query_mol = query_f

    assert len(ref_mol.GetAtoms()) == len(query_mol.GetAtoms())
    #
    coords_pred = query_mol.GetConformer().GetPositions()
    coords_ref = ref_mol.GetConformer().GetPositions()
    # obtain torsion angles
    rotable_bonds = get_torsions([ref_mol])
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(ref_mol, ref_mol.GetConformer(), r, coords_ref)  # coords_pred
    optimized_mol = apply_changes(query_mol, new_dihedrals, rotable_bonds)
    # align
    coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
    R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_ref.T)
    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
    #####
    # apply coords to query
    conf = query_mol.GetConformer()
    for i in range(len(query_mol.GetAtoms())):
        x, y, z = coords_pred_optimized[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    rmsd = rdMolAlign.AlignMol(query_mol, ref_mol)

    query_chiral = Chem.FindMolChiralCenters(query_mol, includeUnassigned=True)
    ref_chiral = Chem.FindMolChiralCenters(ref_mol, includeUnassigned=True)

    return query_mol, round(rmsd, 3)  # , [query_chiral, ref_chiral]


# check their consistency
if __name__ == '__main__':
    root = '/opt/home/revoli/eva/Interformer/energy_output'
    os.makedirs(f"{root}/align", exist_ok=True)
    uff_files = glob.glob(f"{root}/uff/*")
    all = []
    for uff_f in uff_files:
        pdb = os.path.basename(uff_f)[:4]
        # if pdb != '5SAK':
        #   continue
        ref_f = f'{root}/ligand/{pdb}_docked.sdf'
        print(pdb, end=',')
        # random_ref_mol = random_apply_torsion_(ref_f)
        res = align(ref_f, uff_f)  # ref_f
        f = Chem.SDWriter(f'{root}/align/{pdb}_align.sdf')
        align_mol = res[0]
        f.write(align_mol)
        f.close()
        res = [pdb] + list(res[1:])
        all.append(res)
    all.sort(key=lambda x: x[1], reverse=True)
    # merge with docking results
    df = pd.read_csv('/opt/home/revoli/eva/Interformer/energy_output/stat_ligand_reconstructing.csv')
    df = df.groupby('pdb_id').min()
    for item in all:
        t, rmsd = item
        dock_rmsd = df[df.index == t]['rmsd']
        if len(dock_rmsd):
            dock_rmsd = float(dock_rmsd.values[0])
            item.append(dock_rmsd)
        else:
            item.append(-999.)
    print('')
    new_df = pd.DataFrame(all, columns=['pdb_id', 'uff_rmsd', 'dock_rmsd'])
    print(new_df.corr('spearman', numeric_only=True))  # 0.581
    print('done')
