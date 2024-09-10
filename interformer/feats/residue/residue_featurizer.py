import os
import itertools

import numpy as np
import torch
from constant import *
# from anarci import anarci
from sklearn.metrics import pairwise_distances
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import *

amino2abb = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
             'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
             'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
             'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
             'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}


class ReisdueCAFeaturizer(object):

    def __init__(self):
        self.dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
                    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
                    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                    'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
                    'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
                    'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}
        self.allowable_set = ['CO', 'Z', 'Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H', 'C', 'N', 'M', 'D', 'T', 'S',
                              'K',
                              'L', 'Q', 'P']

    def _res2idx(self, res_name):
        res_name = self.dit.get(res_name, 'CO')
        return self.allowable_set.index(res_name)

    def feat_size(self, feat_name=''):
        return 6

    def _normal_tensor(self, x, dim=-1):
        x[:, dim] = (x[:, dim] - x[:, dim].mean()) / x[:, dim].std()

    def _tensor2bins(self, x, dim, bin_edges):
        x[:, dim] = torch.bucketize(x[:, dim].contiguous(), bin_edges)

    def _get_node_feat(self, atoms, chain_id):
        sr = ShrakeRupley()
        if len(atoms):
            atoms[0].parent.parent.atom_to_internal_coordinates(verbose=False)
        ###
        data = []
        for atom in atoms:
            residue = atom.parent
            # sequence relative position
            rel_pos = residue.id[1] / len(residue.parent)
            # sasa
            sr.compute(residue, level="R")
            sasa = residue.sasa
            # torsion angles
            if residue.internal_coord:
                phi = residue.internal_coord.get_angle("phi")
                psi = residue.internal_coord.get_angle("psi")
                phi = phi if phi else 0.
                psi = psi if psi else 0.
            else:
                phi, psi = 0., 0.
            # record
            data.append([self._res2idx(residue.resname), chain_id, rel_pos, sasa, phi, psi])
        return data

    def node_featurize(self, heavy_atoms, light_atoms, ag_atoms):
        feat_size = len(self.allowable_set)
        # AA-feat
        node_feat = []
        node_feat.extend(self._get_node_feat(heavy_atoms, chain_id=0))
        node_feat.extend(self._get_node_feat(light_atoms, chain_id=1))
        node_feat.extend(self._get_node_feat(ag_atoms, chain_id=2))
        node_feat = torch.tensor(node_feat, dtype=torch.float32)
        self._tensor2bins(node_feat, 2, torch.arange(0, 1.0, 0.1))
        self._tensor2bins(node_feat, 3, torch.arange(0, 500, 25))
        self._tensor2bins(node_feat, 4, torch.arange(-180, 180, 20))
        self._tensor2bins(node_feat, 5, torch.arange(-180, 180, 20))
        # xyz
        xyz = []
        xyz.extend([x.coord for x in heavy_atoms])
        xyz.extend([x.coord for x in light_atoms])
        xyz.extend([x.coord for x in ag_atoms])
        xyz = np.array(xyz)
        xyz = torch.tensor(xyz, dtype=torch.float32)
        return node_feat, xyz


def _get_one_chain_CA(mol, chain):
    ca_atoms = []
    if chain in mol:
        for resn in mol[chain].get_residues():
            if 'CA' in resn:  # exclude HET
                ca_atoms.append(resn['CA'])
    return ca_atoms


def _grep_closet_atoms(l_atom, r_atom):
    l_coords = np.array([x.coord for x in l_atom])
    r_coords = np.array([x.coord for x in r_atom])
    dist = pairwise_distances(l_coords, r_coords)
    dist_min_r = np.min(dist, axis=0)
    closest_indices = np.argsort(dist_min_r)[:max_ppi_complex_nodes]
    r_atom = np.array(r_atom)
    r_atom = list(r_atom[closest_indices])
    return r_atom


def _get_CDR_seq(atoms):
    if len(atoms) == 0:
        return []
    # CDR
    cdr1 = [27 - cdr_width, 38 + cdr_width]
    cdr2 = [56 - cdr_width, 65 + cdr_width]
    cdr3 = [105 - cdr_width, 117 + cdr_width]
    # extend cdr
    ###
    seqs = ''.join([amino2abb[x.parent.resname] for x in atoms if x.parent.resname in amino2abb])
    seqs_input = [('seq', seqs)]
    numbering, alignment_details, hit_tables = anarci(seqs_input, scheme="imgt", output=False, assign_germline=False)
    if numbering[0] is None:
        return []
    numbering = numbering[0][0][0]
    cdr_savers = []
    # CDR
    i = 0
    for num in numbering:
        igmt_num = num[0][0]
        if num[-1] != '-':
            if igmt_num >= cdr1[0] and igmt_num <= cdr1[1]:
                cdr_savers.append(i)
            if igmt_num >= cdr2[0] and igmt_num <= cdr2[1]:
                cdr_savers.append(i)
            if igmt_num >= cdr3[0] and igmt_num <= cdr3[1]:
                cdr_savers.append(i)
            # count
            i += 1

    atoms = list(np.array(atoms)[cdr_savers])
    return atoms


def _get_CA_atoms(mol, ligand_chain, receptor_chain):
    # ab_CA
    ab_chains = ligand_chain.split(" | ")
    heavy_atoms = _get_one_chain_CA(mol, ab_chains[0])
    light_atoms = _get_one_chain_CA(mol, ab_chains[1]) if len(ab_chains) == 2 else []
    heavy_atoms = _get_CDR_seq(heavy_atoms)
    light_atoms = _get_CDR_seq(light_atoms)
    ####
    # ag_CA
    ag_chains = receptor_chain
    ag_atoms = []
    for chain in ag_chains:
        ag_atoms.extend(_get_one_chain_CA(mol, chain))
    # Check validation
    if len(ag_atoms) == 0 or len(heavy_atoms) == 0:
        return [], [], []
    # exclude some of the far ag_residue
    ag_atoms = _grep_closet_atoms(heavy_atoms + light_atoms, ag_atoms)
    return heavy_atoms, light_atoms, ag_atoms


def _convert_to_single_emb(x, offset=512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def residue_parser(mol_data, node_featurizer, edge_featurizer, debug=False):
    pdb_f, ligand_chain, receptor_chain = mol_data
    ###
    # 1. Read from BIoPython
    # Special Case for inference
    if isinstance(pdb_f, list):
        if not os.path.exists(pdb_f[0]):
            print(f"# [PPIData] Not Exists File:{pdb_f}")
            return None
        mol1 = PDBParser(QUIET=True).get_structure('ag_st', pdb_f[0])[0]
        mol2 = PDBParser(QUIET=True).get_structure('ab_st', pdb_f[1])[0]
        for child in mol2.child_list:
            mol1.add(child)
        mol = mol1
    else:
        if not os.path.exists(pdb_f):
            print(f"# [PPIData] Not Exists File:{pdb_f}")
            return None
        # Normal case, ag&ab in the same file
        try:
            mol = PDBParser(QUIET=True).get_structure('agab_st', pdb_f)[0]
        except Exception as e:
            print(f"# [PPIData] error loading <- {pdb_f}")
            return None
    ####
    heavy_atoms, light_atoms, ag_atoms = _get_CA_atoms(mol, ligand_chain, receptor_chain)
    ####
    if len(ag_atoms) == 0 or len(heavy_atoms) == 0:
        return None
    if node_featurizer is not None:
        ndata, xyz = node_featurizer.node_featurize(heavy_atoms, light_atoms, ag_atoms)
    #####
    # Packing
    ndata = ndata.to(torch.int32)  # [n, feat_size]
    ndata = _convert_to_single_emb(ndata)  # each feat add 512 offset to it
    xyz = xyz.to(torch.float16)  # [n, 3]
    lens = [len(heavy_atoms) + len(light_atoms), len(ag_atoms)]
    res_lens = [len(heavy_atoms), len(light_atoms), len(ag_atoms)]
    item = {
        'ndata': ndata,
        'xyz': xyz,
        'lens': lens,
        'res_lens': res_lens
    }
    return item
