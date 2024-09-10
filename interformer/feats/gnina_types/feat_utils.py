from rdkit import Chem
import torch
import numpy as np


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


# adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
                binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops + 1):
        extend_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    return extend_mat


def get_las_mask_by_rdkit(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    # add 2-hops
    extend_adj = n_hops_adj(adj, 2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        for i in ring:
            for j in ring:
                if i == j:
                    continue
                else:
                    extend_adj[i][j] += 1
    # turn to mask
    mol_mask = binarize(extend_adj).bool()
    return mol_mask


def get_coords2idx_dict(complex_atoms):
    coords2idx = {}
    for i, a in enumerate(complex_atoms):
        coords2idx[a.coords] = i
    return coords2idx


def coord2idx(coords2idx_dict, coords):
    res = []
    for coord in coords:
        if coord in coords2idx_dict:
            res.append(coords2idx_dict[coord])
    return res


def get_las_mask(l_atoms, r_atoms, plip_mol):
    complex_atoms = l_atoms  # exclude pocket_atoms
    plip_mol = plip_mol.protcomplex
    #
    coords2idx_dict = get_coords2idx_dict(complex_atoms)
    lp = len(complex_atoms)
    adj = torch.zeros([lp, lp], dtype=torch.long)
    # 1. adj matrix for bond-length
    for i, a in enumerate(complex_atoms):
        if hasattr(a, 'neighbors'):  # maybe isolated atom...
            n_idxs = [n.coords for n in a.neighbors]
            n_idxs = coord2idx(coords2idx_dict, n_idxs)
            adj[i, n_idxs] = 1
        # adj[i, i] = 1  # no need self-attend
    # 2. 2-hop extend for bond-angle
    extend_adj = n_hops_adj(adj, 2)
    # 3. SSSR Rings
    # exception
    if not hasattr(plip_mol, 'atom_dict'):
        return None
    for sssr in list(plip_mol.sssr):
        mol_idx = list(sssr._path)
        ring = [np.round(x[1].tolist(), 4).tolist() for x in plip_mol.atom_dict[mol_idx].tolist()]
        ring = [tuple(x) for x in ring]
        ring = coord2idx(coords2idx_dict, ring)
        for i in ring:
            # out of mol_mask
            if i > lp:
                continue
            #
            for j in ring:
                if i == j:
                    continue
                else:
                    extend_adj[i][j] += 1
    # 4. turn into mask
    mol_mask = binarize(extend_adj).bool()
    # 5. self-distance-mask
    mol_mask = mol_mask + torch.eye(mol_mask.size(0)).bool()
    # mol_mask = ~mol_mask  # reverse LAS, 1=las, 0=non-las
    return mol_mask
