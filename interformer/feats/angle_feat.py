import numpy as np
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.metrics import pairwise_distances

__all__ = ['bond_angle', 'interaction_angle', 'partial_charge']


def test_angle_converter():
    angle = 50.
    angle_pi = angle * np.pi / 180.
    x = np.sin(angle_pi)
    y = np.cos(angle_pi)
    res = np.arctan2(x, y)
    res_angle = res * 180. / np.pi


def cal_Hbond_angle(coords, N):
    # get the 2 closet neighbors
    paris = pairwise_distances(coords, coords)
    paris_sorted = np.argsort(paris)
    three_atoms = []
    for value in paris_sorted:
        three_atoms.append([value[1], value[0], value[2]])
    three_atoms = np.array(three_atoms)
    # calculate bond angle
    three_coords = coords[three_atoms]
    ab = three_coords[:, 0, :] - three_coords[:, 1, :]
    cb = three_coords[:, 2, :] - three_coords[:, 1, :]
    b_coord = three_coords[:, 1, :]
    # adding b_coords
    hbond_vec = ab + cb
    h_coord = b_coord + hbond_vec
    # 3. found the closest other side's atoms
    ligands_coords, amino_coords = coords[:N], coords[N:]
    complex_pairs = pairwise_distances(ligands_coords, amino_coords)


######## Useful functions ############

def cal_angle_by_3coords(three_coords):
    ab = three_coords[:, 0, :] - three_coords[:, 1, :]
    cb = three_coords[:, 2, :] - three_coords[:, 1, :]
    cos = np.sum(ab * cb, axis=-1) / ((np.linalg.norm(ab, axis=1) * np.linalg.norm(cb, axis=1)) + 1e-9)
    angle = np.arccos(np.clip(cos, -1., 1.)) / np.pi
    # if the angle is 0.5 meaning the cos is zero, force to be zero as well
    angle = np.where(angle == 0.5, 0., angle)
    return angle


def get_angle(idx, closest_opposite_atom, complex, coords):
    N = len(idx)
    if N == 0:
        return []
    # 1. loop over idx
    abc_coords = []
    for i in idx:
        atom = complex.GetAtomWithIdx(i)
        nbr_coords = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            nbr_coords.append(coords[nbr_idx])
            # debug
            # print(i, '->', nbr_idx)
        abc_coords.append([coords[i], nbr_coords, closest_opposite_atom[i]])
    # 2. generate angle feat
    # maximum 5 bonds
    final_angle = np.zeros([N, 4], dtype=np.float64)
    for i in range(4):
        three_coords = np.array(
            [[x[0], x[1][i], x[2]] if i < len(x[1]) else np.zeros([3, 3], dtype=np.float64) for x in abc_coords])
        angle = cal_angle_by_3coords(three_coords)
        final_angle[:, i] = angle
    return final_angle


##########Outer Fucntion#####
def bond_angle(coords):
    # exception, if there is less than 3 atoms...
    if len(coords) < 3:
        return torch.zeros(len(coords), 1).float().view(-1, 1)
    # get the 2 closet neighbors
    paris = pairwise_distances(coords, coords)
    paris_sorted = np.argsort(paris)
    three_atoms = []
    for value in paris_sorted:
        three_atoms.append([value[1], value[0], value[2]])
    three_atoms = np.array(three_atoms)
    # calculate bond angle
    three_coords = coords[three_atoms]
    ab = three_coords[:, 0, :] - three_coords[:, 1, :]
    cb = three_coords[:, 2, :] - three_coords[:, 1, :]
    cos = np.sum(ab * cb, axis=-1) / ((np.linalg.norm(ab, axis=1) * np.linalg.norm(cb, axis=1)) + 1e-9)
    angle = np.arccos(np.clip(cos, -1., 1.)) / np.pi
    # Degree for review only
    # degree = np.degrees(angle)
    angle = torch.tensor(angle).float().view(-1, 1)
    return angle


def interaction_angle(complex, ligand_idx, amino_idx):
    LN = len(ligand_idx)
    coords = complex.GetConformer().GetPositions()
    ligands_coords, amino_coords = coords[:LN], coords[LN:]
    complex_pairs = pairwise_distances(ligands_coords, amino_coords)
    # get closest amino
    paris_sorted = np.argsort(complex_pairs)
    closest_amino_atom = coords[paris_sorted[:, 0] + LN]
    # get closest ligand
    closest_ligand_atom = coords[np.argsort(np.transpose(complex_pairs))[:, 0]]
    closest_ligand_atom = np.concatenate([closest_amino_atom, closest_ligand_atom], axis=0)
    ##########
    ligand_angle = get_angle(ligand_idx, closest_amino_atom, complex, coords)
    amino_angle = get_angle(amino_idx, closest_ligand_atom, complex, coords)
    if len(ligand_angle) and len(amino_angle):
        all_angle = np.concatenate([ligand_angle, amino_angle], axis=0)
    else:
        all_angle = ligand_angle
    all_angle = torch.tensor(all_angle, dtype=torch.float)
    return all_angle


def partial_charge(complex, v):
    try:
        Chem.SanitizeMol(complex)
    except:
        # if can't sanitize the mol just return zero
        return torch.zeros([len(v), 1], dtype=torch.float)
    complex_withH = Chem.AddHs(complex)
    mmff_pro = AllChem.MMFFGetMoleculeProperties(complex_withH, mmffVariant='MMFF94s')
    if mmff_pro is None:
        return torch.zeros([len(v), 1], dtype=torch.float)
    ref_partial_charge = torch.tensor([mmff_pro.GetMMFFPartialCharge(i) for i in v], dtype=torch.float)
    ref_partial_charge = ref_partial_charge.view(-1, 1)
    ###
    # debug
    # tmp = []
    # for atom in complex.GetAtoms():
    #   tmp.append(atom.GetSymbol())
    # formal_charges = [atom.GetFormalCharge() for atom in complex.GetAtoms()]
    # formal_charges = [formal_charges[i] for i in v]
    # tmp = [tmp[i] for i in v]
    # print(tmp)
    # print('-' * 100)
    # tmp = []
    # for atom in complex_withH.GetAtoms():
    #   tmp.append(atom.GetSymbol())
    # print(tmp)
    # They are the same formal charge
    # ref_formal_charge = torch.tensor([mmff_pro.GetMMFFFormalCharge(i) for i in v], dtype=torch.float)
    # print(ref_formal_charge)
    # print(formal_charges)
    #
    return ref_partial_charge


if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    pdb = Chem.MolFromPDBFile('/opt/home/revoli/data_worker/v2019-docking/pocket/1h00_pocket.pdb')
    sdf = Chem.SDMolSupplier('/opt/home/revoli/data_worker/v2019-docking/ligands/1h00_docked.sdf')[0]
    complex = rdmolops.CombineMols(sdf, pdb)
    N = len(sdf.GetAtoms())
    # get the 2 closet neighbors
    coords = complex.GetConformer().GetPositions()
    angle = bond_angle(coords)
    # get 2ligand->1 amino angle and 2 amino -> 1 ligand angle
    angle2 = interaction_angle(complex, list(range(N)), [40, 44, 55, 77])
    # partial charge
    partial_charge_feat = partial_charge(complex, list(range(N)))
