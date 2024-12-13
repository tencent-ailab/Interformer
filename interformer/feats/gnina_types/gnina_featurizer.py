import os.path
import warnings
from collections import defaultdict

import numpy as np
import torch
from openbabel import openbabel, pybel
from plip.structure.preparation import PDBComplex
from rdkit import Chem
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from constant import *
from feats.gnina_types.obabel_api import clean_pdb_intersection_code, rm_water_from_pdb

warnings.filterwarnings("ignore", category=FutureWarning)


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class Info:
    """Data structure to hold atom type data"""

    def __init__(
        self,
        sm,
        smina_name,
        adname,
        anum,
        ad_radius,
        ad_depth,
        ad_solvation,
        ad_volume,
        covalent_radius,
        xs_radius,
        xs_hydrophobe,
        xs_donor,
        xs_acceptor,
        ad_heteroatom,
    ):
        self.sm = sm
        self.smina_name = smina_name
        self.adname = adname
        self.anum = anum
        self.ad_radius = ad_radius
        self.ad_depth = ad_depth
        self.ad_solvation = ad_solvation
        self.ad_volume = ad_volume
        self.covalent_radius = covalent_radius
        self.xs_radius = xs_radius
        self.xs_hydrophobe = xs_hydrophobe
        self.xs_donor = xs_donor
        self.xs_acceptor = xs_acceptor
        self.ad_heteroatom = ad_heteroatom


class PLIPAtomFeaturizer(object):
    def __init__(self, plip_feat=False):
        self.plip_feat = plip_feat
        self.amino_type = False
        #
        self.non_ad_metal_names = [
            "Cu",
            "Fe",
            "Na",
            "K",
            "Hg",
            "Co",
            "U",
            "Cd",
            "Ni",
            "Si",
        ]
        self.atom_equivalence_data = [("Se", "S")]
        self.atom_type_data = [
            Info(
                "Hydrogen",
                "Hydrogen",
                "H",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "PolarHydrogen",
                "PolarHydrogen",
                "HD",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSHydrophobe",
                "AliphaticCarbonXSHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSNonHydrophobe",
                "AliphaticCarbonXSNonHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSHydrophobe",
                "AromaticCarbonXSHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSNonHydrophobe",
                "AromaticCarbonXSNonHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "Nitrogen",
                "Nitrogen",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonor",
                "NitrogenXSDonor",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonorAcceptor",
                "NitrogenXSDonorAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "NitrogenXSAcceptor",
                "NitrogenXSAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Oxygen",
                "Oxygen",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "OxygenXSDonor",
                "OxygenXSDonor",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "OxygenXSDonorAcceptor",
                "OxygenXSDonorAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "OxygenXSAcceptor",
                "OxygenXSAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Sulfur",
                "Sulfur",
                "S",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "SulfurAcceptor",
                "SulfurAcceptor",
                "SA",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Phosphorus",
                "Phosphorus",
                "P",
                15,
                2.100000,
                0.200000,
                -0.001100,
                38.792400,
                1.060000,
                2.100000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Fluorine",
                "Fluorine",
                "F",
                9,
                1.545000,
                0.080000,
                -0.001100,
                15.448000,
                0.710000,
                1.500000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Chlorine",
                "Chlorine",
                "Cl",
                17,
                2.045000,
                0.276000,
                -0.001100,
                35.823500,
                0.990000,
                1.800000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Bromine",
                "Bromine",
                "Br",
                35,
                2.165000,
                0.389000,
                -0.001100,
                42.566100,
                1.140000,
                2.000000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Iodine",
                "Iodine",
                "I",
                53,
                2.360000,
                0.550000,
                -0.001100,
                55.058500,
                1.330000,
                2.200000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Magnesium",
                "Magnesium",
                "Mg",
                12,
                0.650000,
                0.875000,
                -0.001100,
                1.560000,
                1.300000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Manganese",
                "Manganese",
                "Mn",
                25,
                0.650000,
                0.875000,
                -0.001100,
                2.140000,
                1.390000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Zinc",
                "Zinc",
                "Zn",
                30,
                0.740000,
                0.550000,
                -0.001100,
                1.700000,
                1.310000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Calcium",
                "Calcium",
                "Ca",
                20,
                0.990000,
                0.550000,
                -0.001100,
                2.770000,
                1.740000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Iron",
                "Iron",
                "Fe",
                26,
                0.650000,
                0.010000,
                -0.001100,
                1.840000,
                1.250000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "GenericMetal",
                "GenericMetal",
                "M",
                0,
                1.200000,
                0.000000,
                -0.001100,
                22.449300,
                1.750000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            # note AD4 doesn't have boron, so copying from carbon
            Info(
                "Boron",
                "Boron",
                "B",
                5,
                2.04,
                0.180000,
                -0.0011,
                12.052,
                0.90,
                1.920000,
                True,
                False,
                False,
                False,
            ),
        ]
        self.atom_types = [info.sm for info in self.atom_type_data]
        self.type_map = self.get_type_map()

        self.atom_types = ['Empty Atom'] + [info.sm for info in self.atom_type_data]

    def get_type_map(self):
        """Original author: Constantin Schneider"""
        types = [
            ['AliphaticCarbonXSHydrophobe'],
            ['AliphaticCarbonXSNonHydrophobe'],
            ['AromaticCarbonXSHydrophobe'],
            ['AromaticCarbonXSNonHydrophobe'],
            ['Nitrogen', 'NitrogenXSAcceptor'],
            ['NitrogenXSDonor', 'NitrogenXSDonorAcceptor'],
            ['Oxygen', 'OxygenXSAcceptor'],
            ['OxygenXSDonor', 'OxygenXSDonorAcceptor'],
            ['Sulfur', 'SulfurAcceptor'],
            ['Phosphorus']
        ]
        out_dict = {}
        generic = []
        for i, element_name in enumerate(self.atom_types):
            for types_list in types:
                if element_name in types_list:
                    out_dict[i] = types.index(types_list)
                    break
            if i not in out_dict.keys():
                generic.append(i)

        generic_type = len(types)
        for other_type in generic:
            out_dict[other_type] = generic_type
        return out_dict

    def string_to_smina_type(self, string: str):
        """Convert string type to smina type.

        Original author: Constantin schneider

        Args:
            string (str): string type
        Returns:
            string: smina type
        """
        if len(string) <= 2:
            for type_info in self.atom_type_data:
                # convert ad names to smina types
                if string == type_info.adname:
                    return type_info.sm
            # find equivalent atoms
            for i in self.atom_equivalence_data:
                if string == i[0]:
                    return self.string_to_smina_type(i[1])
            # generic metal
            if string in self.non_ad_metal_names:
                return "GenericMetal"
            # if nothing else found --> generic metal
            return "GenericMetal"

        else:
            # assume it's smina name
            for type_info in self.atom_type_data:
                if string == type_info.smina_name:
                    return type_info.sm
            # if nothing else found, return numtypes
            # technically not necessary to call this numtypes,
            # but including this here to make it equivalent to the cpp code
            return "NumTypes"

    @staticmethod
    def adjust_smina_type(t, h_bonded, hetero_bonded):
        """Original author: Constantin schneider"""
        if t in ('AliphaticCarbonXSNonHydrophobe',
                 'AliphaticCarbonXSHydrophobe'):  # C_C_C_P,
            if hetero_bonded:
                return 'AliphaticCarbonXSNonHydrophobe'
            else:
                return 'AliphaticCarbonXSHydrophobe'
        elif t in ('AromaticCarbonXSNonHydrophobe',
                   'AromaticCarbonXSHydrophobe'):  # C_A_C_P,
            if hetero_bonded:
                return 'AromaticCarbonXSNonHydrophobe'
            else:
                return 'AromaticCarbonXSHydrophobe'
        elif t in ('Nitrogen', 'NitogenXSDonor'):
            # N_N_N_P, no hydrogen bonding
            if h_bonded:
                return 'NitrogenXSDonor'
            else:
                return 'Nitrogen'
        elif t in ('NitrogenXSAcceptor', 'NitrogenXSDonorAcceptor'):
            # N_NA_N_A, also considered an acceptor by autodock
            if h_bonded:
                return 'NitrogenXSDonorAcceptor'
            else:
                return 'NitrogenXSAcceptor'
        elif t in ('Oxygen', 'OxygenXSDonor'):  # O_O_O_P,
            if h_bonded:
                return 'OxygenXSDonor'
            else:
                return 'Oxygen'
        elif t in ('OxygenXSAcceptor', 'OxygenXSDonorAcceptor'):
            # O_OA_O_A, also an autodock acceptor
            if h_bonded:
                return 'OxygenXSDonorAcceptor'
            else:
                return 'OxygenXSAcceptor'
        else:
            return t

    def obatom_to_smina_type(self, ob_atom):
        """Original author: Constantin schneider"""
        atomic_number = ob_atom.atomicnum
        num_to_name = {1: 'HD', 6: 'A', 7: 'NA', 8: 'OA', 16: 'SA'}

        # Default fn returns True, otherwise inspect atom properties
        condition_fns = defaultdict(lambda: lambda: True)
        condition_fns.update({
            6: ob_atom.OBAtom.IsAromatic,
            7: ob_atom.OBAtom.IsHbondAcceptor,
            16: ob_atom.OBAtom.IsHbondAcceptor
        })

        # Get symbol
        ename = openbabel.GetSymbol(atomic_number)
        # Do we need to adjust symbol?
        if condition_fns[atomic_number]():
            ename = num_to_name.get(atomic_number, ename)

        atype = self.string_to_smina_type(ename)

        h_bonded = False
        hetero_bonded = False
        for neighbour in openbabel.OBAtomAtomIter(ob_atom.OBAtom):
            if neighbour.GetAtomicNum() == 1:
                h_bonded = True
            elif neighbour.GetAtomicNum() != 6:
                hetero_bonded = True

        final_atom_type = self.adjust_smina_type(atype, h_bonded, hetero_bonded)
        return final_atom_type

    def amino2num(self, res_name):
        dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
               'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
               'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
               'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
               'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
               'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}
        # 1-CO=Cofactor, 2-Z=LIG
        allowable_set = ['CO', 'Z', 'Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H', 'C', 'N', 'M', 'D', 'T', 'S', 'K',
                         'L', 'Q', 'P']
        res_name = dit.get(res_name, 'CO')
        return allowable_set.index(res_name)

    def feat_size(self, feat_name=None):
        # 1. gnina-type, 2. amino-types
        return 1

    def __call__(self, l_atoms, r_atoms):
        processed_features = []
        for a in l_atoms:
            gnina_type = self.obatom_to_smina_type(a)
            l_feats = [self.atom_types.index(gnina_type)]
            processed_features.append(l_feats)
        for i, a in enumerate(r_atoms):
            gnina_type = self.obatom_to_smina_type(a)
            r_feats = [self.atom_types.index(gnina_type) + len(self.atom_types)]
            processed_features.append(r_feats)
        #
        processed_features = torch.tensor(processed_features, dtype=torch.float32)
        return processed_features


class PairTypeFeat:

    def __init__(self):
        feat_indices = {}
        # (29 * 28) / 2 permutations
        r = 0
        for i in range(0, 30):
            for j in range(0, 30):
                f_str = f'{j}_{i}' if i > j else f'{i}_{j}'
                if f_str not in feat_indices:
                    feat_indices[f_str] = r
                    r += 1
                # fix 0 to be 0
                if i == 0 or j == 0:
                    feat_indices[f_str] = 0
        self.feat_indices = feat_indices

    @staticmethod
    def lower(idx):
        if idx > 29:
            idx -= 29
        return idx

    def AtomType2PairType(self, x):
        N, _ = x.size()
        pair_type = torch.zeros(N, N, 1, dtype=torch.int)
        for i, atom_i in enumerate(x):
            for j, atom_j in enumerate(x):
                atom_i, atom_j = self.lower(int(atom_i)), self.lower(int(atom_j))
                f_str = f'{atom_j}_{atom_i}' if atom_i > atom_j else f'{atom_i}_{atom_j}'
                pair_type[i, j] = self.feat_indices[f_str]
        return pair_type


class PLIPEdgeFeaturizer(object):
    def __init__(self, covalent_bond=False, interaction_angle=False):
        self.plip_node_feat_index = ['HACC', 'HDON', 'PISTACK', 'PICATION_RING', 'PICATION_POS', 'PICATION_NEG',
                                     'HYDROPH',
                                     'SALT_POS', 'SALT_NEG', 'XBOND_ACC', 'XBOND_DON']
        self.plip_edge_feat_index = ['no', 'hbond', 'saltbridge', 'halogenbond', 'pication', 'pistack',
                                     'hydroph_interaction']
        self.covalent_bond = covalent_bond
        self.feat_dim = 1
        if covalent_bond:
            self.feat_dim += 1
        self.interaction_angle = interaction_angle
        # pair-type
        # self.pair_type = True
        # self.pair_type_cls = PairTypeFeat()

    def get_pair_type(self, x, edata):
        pair_type = self.pair_type_cls.AtomType2PairType(x)
        edata = torch.cat([edata, pair_type], dim=-1)
        return edata

    def feat_size(self):
        return self.feat_dim

    def idx2l_r_atoms(self, l_atoms, r_atoms):
        idx2atoms_index = {}
        for i, a in enumerate(l_atoms + r_atoms):
            idx2atoms_index[a.idx] = i
        return idx2atoms_index

    def append_covalent_bond(self, idx2a, l_atoms, r_atoms, edge_feats):
        all_atoms = l_atoms + r_atoms
        new_feats = torch.zeros_like(edge_feats)
        for a in all_atoms:
            a_idx = idx2a[a.idx]
            for bond in a.bonds:
                n_atom = bond.atoms[1]
                # make-sure bond should be in the complex graph
                if n_atom.idx in idx2a:
                    n_idx = idx2a[n_atom.idx]
                    new_feats[a_idx, n_idx] = bond.order
        edge_feats = torch.cat([edge_feats, new_feats], dim=-1)
        return edge_feats

    def process_inter_angles(self, l_atoms, r_atoms, plip_interaction):
        plip_interaction = plip_interaction[list(plip_interaction.keys())[0]]
        idx2a = self.idx2l_r_atoms(l_atoms, r_atoms)
        N = len(l_atoms) + len(r_atoms)
        angle_feats = torch.zeros([N, 1], dtype=torch.float32)

        def assign_bond_feat(a_idx, b_idx, angle):
            # out of amino_atoms, just skip them
            if a_idx not in idx2a or b_idx not in idx2a:
                return
            a_idx, b_idx = idx2a[a_idx], idx2a[b_idx]
            angle = angle / 180.
            angle_feats[a_idx] = angle
            angle_feats[b_idx] = angle

        # run
        for inter in plip_interaction.all_itypes:
            inter_name = type(inter).__name__
            if 'hbond' == inter_name:
                # H-bond
                A_idx, D_idx = inter.a.idx, inter.d.idx
                assign_bond_feat(A_idx, D_idx, inter.angle)
            elif 'saltbridge' == inter_name:
                # Salt-Bridge
                pass
            elif 'halogenbond' == inter_name:
                # X-Bond
                pass
            elif 'pication' == inter_name:
                # Pi-Cation, No angle for now
                pass
            elif 'pistack' == inter_name:
                # Pi-Stack
                ligand_idx = [x.idx for x in inter.ligandring.atoms]
                pocket_idx = [x.idx for x in inter.proteinring.atoms]
                for a in ligand_idx:
                    for b in pocket_idx:
                        assign_bond_feat(a, b, inter.angle)
            elif 'hydroph_interaction' == inter_name:
                # Hydrophobe
                pass
            # ignored
            elif 'waterbridge' == inter_name:
                pass
        return angle_feats

    def __call__(self, l_atoms, r_atoms, plip_interaction):
        plip_interaction = plip_interaction[list(plip_interaction.keys())[0]]
        idx2a = self.idx2l_r_atoms(l_atoms, r_atoms)
        N = len(l_atoms) + len(r_atoms)
        edge_feats = torch.zeros([N, N, 1], dtype=torch.int32)

        def assign_bond_feat(a_idx, b_idx, inter_name):
            # out of amino_atoms, just skip them
            if a_idx not in idx2a or b_idx not in idx2a:
                return
            a_idx, b_idx = idx2a[a_idx], idx2a[b_idx]
            edge_feats[a_idx, b_idx] = self.plip_edge_feat_index.index(inter_name)
            edge_feats[b_idx, a_idx] = self.plip_edge_feat_index.index(inter_name)

        # run
        for inter in plip_interaction.all_itypes:
            inter_name = type(inter).__name__
            if 'hbond' == inter_name:
                # H-bond
                A_idx, D_idx = inter.a.idx, inter.d.idx
                assign_bond_feat(A_idx, D_idx, inter_name)
            elif 'saltbridge' == inter_name:
                # Salt-Bridge
                neg_indices = [x.idx for x in inter.negative.atoms]
                pos_indices = [x.idx for x in inter.positive.atoms]
                for neg in neg_indices:
                    for pos in pos_indices:
                        assign_bond_feat(neg, pos, inter_name)
            elif 'halogenbond' == inter_name:
                # X-Bond
                hal_acc = inter.acc[0].idx
                hal_don = inter.don[0].idx
                assign_bond_feat(hal_acc, hal_don, inter_name)
            elif 'pication' == inter_name:
                # Pi-Cation
                ring_idx = [x.idx for x in inter.ring.atoms]
                charge_idx = inter.charge.atoms[0].idx
                for ring in ring_idx:
                    assign_bond_feat(ring, charge_idx, inter_name)
            elif 'pistack' == inter_name:
                # Pi-Stack
                ligand_idx = [x.idx for x in inter.ligandring.atoms]
                pocket_idx = [x.idx for x in inter.proteinring.atoms]
                for a in ligand_idx:
                    for b in pocket_idx:
                        assign_bond_feat(a, b, inter_name)
            elif 'hydroph_interaction' == inter_name:
                # Hydrophobe
                res_atom = inter.bsatom.idx
                ligand_atom = inter.ligatom.idx
                assign_bond_feat(res_atom, ligand_atom, inter_name)
            # ignored
            elif 'waterbridge' == inter_name:
                pass
        # statis
        # np.histogram(edge_feats, bins=len(self.plip_edge_feat_index))
        # append covalent
        if self.covalent_bond:
            edge_feats = self.append_covalent_bond(idx2a, l_atoms, r_atoms, edge_feats)
        return edge_feats


def get_computed_info(l_atoms, r_atoms, plip_mol):
    processed_features = [[] for _ in range(len(l_atoms) + len(r_atoms))]
    total_num_feat = 2
    # 0. degree
    for i, a in enumerate(l_atoms + r_atoms):
        d = a.heavydegree
        processed_features[i].append(d)
    # 1. partial charge
    # for a in l_atoms + r_atoms:
    #   processed_features.append(a.partialcharge)
    # processed_features = torch.tensor(processed_features)
    # processed_features = processed_features.view(-1, 1)
    # 2. bond angles  # TODO: Bond-Angle(3 atoms form an angle), a little bit difficult to do
    for i, a in enumerate(l_atoms + r_atoms):
        # Using official average bond angle(including hydrogen)
        avg_bond_angle = a.OBAtom.AverageBondAngle()
        processed_features[i].append(np.cos(avg_bond_angle))
        # Self-Calculation, find two heavy-atom-neighbors
        # d = a.heavydegree
        # at least has two bonds
        # if d > 1:
        #   n_idx = [n.for n in a.neighbors]
    processed_features = torch.tensor(processed_features, dtype=torch.float32)
    processed_features = processed_features.view(-1, total_num_feat)
    return processed_features


####

def get_coords(atoms):
    coords = []
    for a in atoms:
        coords.append(a.coords)
    return torch.tensor(coords).float()


def split_ligand_pocket(plip_mol):
    ligand_atoms, pocket_atoms = [], []
    for i, a in plip_mol.atoms.items():
        # skip those hydrogen
        if a.atomicnum != 1:
            if a.residue.name == 'LIG':
                ligand_atoms.append(a)
            else:
                pocket_atoms.append(a)
    return ligand_atoms, pocket_atoms


def get_closest_amino(plip_mol, max_amino_nodes=400):
    l_atoms, r_atoms = split_ligand_pocket(plip_mol)
    # can't find pocket
    if len(r_atoms) == 0:
        print(f"[get_closest_amino] No pocket atoms.")
        return None, None, None
    if len(l_atoms) == 0:
        print(f"[get_closest_amino] No ligand atoms.")
        return None, None, None
    # Filter those r_coords is larger than subgraph0_dist
    l_coords = get_coords(l_atoms)
    r_coords = get_coords(r_atoms)
    D = torch.cdist(r_coords, l_coords)
    # get closest aminos
    closest_amino_D = D.min(-1)[0]
    farest_D_mask = (closest_amino_D < subgraph0_dist).bool()
    if len(farest_D_mask) > max_amino_nodes:  #
        # print(f"MaxAminoAtoms-{len(farest_D_mask)}", end=',')  # showing all the screen, removed
        cloests_indices = closest_amino_D.sort()[1][max_amino_nodes:]
        farest_D_mask[cloests_indices] = 0
    r_atoms = [r_atoms[i] for i, x in enumerate(farest_D_mask) if x]
    # xyz
    r_coords = get_coords(r_atoms)
    xyz = torch.cat([l_coords, r_coords], dim=0)
    ###
    # For Docking wanted atoms
    ref_atoms_indices = [i for i in range(len(l_coords))]  # exclude this from complex pdb
    pocket_atoms_indices = (farest_D_mask.nonzero() + len(l_coords)).view(-1).tolist()
    wanted_atoms_indices = pocket_atoms_indices
    return l_atoms, r_atoms, wanted_atoms_indices


def plip_preprocess(complex_str):
    # PLIP
    plip_handler = PDBComplex()
    plip_handler.load_pdb(complex_str, as_string=True)
    for ligand in plip_handler.ligands:
        plip_handler.characterize_complex(ligand)
    return plip_handler


def output_complex(energy_output_folder, pdb, uff_ligand, ligand, complex_str, wai):
    # create folder
    os.makedirs(f'{energy_output_folder}/complex', exist_ok=True)
    os.makedirs(f'{energy_output_folder}/ligand', exist_ok=True)
    os.makedirs(f'{energy_output_folder}/uff', exist_ok=True)
    #
    complex = []
    complex_context = complex_str.split('\n')
    for line_id in wai:
        complex.append(complex_context[line_id + 2])
    assert len(complex) == len(wai)  # should be the same len
    complex = complex_context[:2] + complex
    # write complex
    complex = [x.strip() + '\n' for x in complex]
    f = open(f'{energy_output_folder}/complex/{pdb}_complex.pdb', 'w')
    f.writelines(complex)
    f.close()
    # copy ligand sdf
    if ligand is not None:
        writer = Chem.SDWriter(f'{energy_output_folder}/ligand/{pdb}_docked.sdf')
        writer.write(ligand)
        writer.close()
    # copy uff ligand
    # TODO: it only copy once [when VS mode is on, uff file should contain multiple ligands]
    if uff_ligand is not None:
        writer = Chem.SDWriter(f'{energy_output_folder}/uff/{pdb}_uff.sdf')
        writer.write(uff_ligand)
        writer.close()


def get_D_rdkit(mol):
    xyz = np.array(mol.GetConformer().GetPositions())
    D = torch.tensor(cdist(xyz, xyz), dtype=torch.float)  # [n, n]
    return D


def get_xyz_by_rdkit(mol):
    # do not get hydrogen atom
    ids = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() != 1:
            ids.append(i)
    #
    coords = mol.GetConformer().GetPositions()
    coords = coords[ids]
    return torch.tensor(coords)


def exit_condition(mol_data, l_atoms, r_atoms, pdb):
    if r_atoms is None or l_atoms is None:
        print(f"# No Pocket or ligand. Pocket<-{r_atoms}, ligand<-{l_atoms}", end=',')
        return None
    if len(l_atoms) > max_ligand_atoms:
        print(f'# Number of Ligand atoms exceeds maximum value of {max_ligand_atoms}.')
        return None
    if len(l_atoms) < min_ligand_atoms:
        print(f'# Number of Ligand atoms is too less than {min_ligand_atoms}.')
        return None
    if 'uff_ligand' in mol_data:
        if mol_data['uff_ligand'] is None:
            print("# Can't read uff_ligand.")
            return None
    return True


def merge_pocket_ligand(pocket, ligand, pdb):
    try:
        complex = Chem.rdmolops.CombineMols(ligand, pocket)
        complex_str = Chem.MolToPDBBlock(complex)
    except Exception as e:
        print(f'Cofactor is wrong->{e}, {pdb}')
        return None
    #
    complex_str = clean_pdb_intersection_code(complex_str)
    complex_str = rm_water_from_pdb(complex_str)
    obabel_complex = pybel.readstring('pdb', complex_str)
    complex_str = obabel_complex.write('pdb')
    return complex_str


def rdkit2obabel_sdf(ligand):
    sdf_string = Chem.MolToMolBlock(ligand)
    mol = pybel.readstring('sdf', sdf_string)
    mol = mol.clone  # it is extremely important, if you don't clone,
    # and access its property, it will throw segmentation fault
    l_atoms = []
    for a in mol.atoms:
        if a.atomicnum != 1:
            l_atoms.append(a)
    return l_atoms


def get_gt_xyz(l_atoms, r_atoms):
    l_coords = get_coords(l_atoms)
    r_coords = get_coords(r_atoms)
    xyz = torch.cat([l_coords, r_coords], dim=0)
    return xyz


def obabel_mol_parser(mol_data, node_featurizer, edge_featurizer, args):
    complex_str = mol_data['complex_str']
    if complex_str is None:
        return None
    ######
    # PLIP
    plip_mol = plip_preprocess(complex_str)
    # Get Closest amino atoms
    max_amino_nodes = affinity_max_amino_nodes if args['affinity_pre'] else energy_max_amino_nodes
    l_atoms, r_atoms, wai = get_closest_amino(plip_mol, max_amino_nodes=max_amino_nodes)
    condition = exit_condition(mol_data, l_atoms, r_atoms, mol_data['pdb'])
    if condition is None:
        print(f"# [obabel_mol_parser]: Error {mol_data['pdb']}", end=',')
        return None
    ###
    # uff_as_ligand
    if 'uff_as_ligand' in mol_data and mol_data['uff_as_ligand']:
        mol = pybel.readstring('sdf', Chem.MolToMolBlock(mol_data['uff_ligand'])).clone
        l_atoms = [a for a in mol.atoms if a.atomicnum != 1]
    ##
    # Feat Area
    ndata, edata = None, None
    if node_featurizer is not None:
        ndata = node_featurizer(l_atoms, r_atoms)
    if edge_featurizer is not None:
        edata = edge_featurizer(l_atoms, r_atoms, plip_mol.interaction_sets)
    ##
    # Getting UFF's xyz
    if 'uff_ligand' in mol_data:
        uff_xyz = get_xyz_by_rdkit(mol_data['uff_ligand'])
        uff_xyz = uff_xyz.to(torch.float16)
    else:
        uff_xyz = None
    ##########
    # Packing
    ndata = ndata.to(torch.int32)
    gt_xyz = get_gt_xyz(l_atoms, r_atoms).to(torch.float16)
    # item packing
    lens = [len(l_atoms), len(r_atoms)]
    item = {
        'ndata': ndata,
        'edata': edata,
        'gt_xyz': gt_xyz,
        'uff_xyz': uff_xyz,
        'lens': lens,
        'pdb': mol_data['pdb'],
        'target': mol_data['target']
    }
    ## Vis
    if args['energy_output_folder']:
        uff_ligand = mol_data['uff_ligand'] if 'uff_ligand' in mol_data else None
        output_complex(args['energy_output_folder'], mol_data['pdb'], uff_ligand, mol_data['ligand'], complex_str, wai)
    return item


#########
# PPI
def get_ligand_rec(mol, ligand_chain, receptor_chain):
    l_atoms = [atom for atom in mol.atoms if atom.residue.OBResidue.GetChain() in ligand_chain]
    r_atoms = [atom for atom in mol.atoms if atom.residue.OBResidue.GetChain() in receptor_chain]
    # filter H atom
    l_atoms = np.array([atom for atom in l_atoms if atom.atomicnum != 1])
    r_atoms = np.array([atom for atom in r_atoms if atom.atomicnum != 1])
    return l_atoms, r_atoms


def get_closest_r_atoms(l_atoms, r_atoms):
    if len(l_atoms) + len(r_atoms) < max_ppi_complex_nodes:
        return l_atoms, r_atoms
    l_coords = np.array([x.coords for x in l_atoms])
    r_coords = np.array([x.coords for x in r_atoms])
    dist = pairwise_distances(l_coords, r_coords)
    dist_min_r = np.min(dist, axis=0)
    dist_min_l = np.min(dist, axis=1)
    dist_min = np.concatenate([dist_min_l, dist_min_r])
    closest_indices = np.argsort(dist_min)[:max_ppi_complex_nodes]
    l_indices = [i for i in closest_indices if i < len(l_atoms)]
    r_indices = [i - len(l_atoms) for i in closest_indices if i >= len(l_atoms)]
    l_atoms, r_atoms = l_atoms[l_indices], r_atoms[r_indices]
    return l_atoms, r_atoms


def get_interface_atoms(l_atoms, r_atoms, contact_th=5.3):
    l_coords = np.array([x.coords for x in l_atoms])
    r_coords = np.array([x.coords for x in r_atoms])
    dist = pairwise_distances(l_coords, r_coords)
    interface_indices = np.where(dist < contact_th)
    inter_ligand_indices, inter_rec_indices = np.unique(interface_indices[0]), np.unique(interface_indices[1])
    l_atoms, r_atoms = l_atoms[inter_ligand_indices], r_atoms[inter_rec_indices]
    # get cloest r_atoms
    l_atoms, r_atoms = get_closest_r_atoms(l_atoms, r_atoms)
    # cat l_atoms, r_atoms's xyz
    xyz = torch.tensor([x.coords for x in np.concatenate([l_atoms, r_atoms])])
    return l_atoms, r_atoms, xyz


def obabel_ppi_parser(mol_data, node_featurizer, edge_featurizer, debug=False):
    pdb_f, ligand_chain, receptor_chain = mol_data
    ###
    if not os.path.exists(pdb_f):
        print(f"# [PPIData] Not Exists File:{pdb_f}")
        return None
    # 1. Read from PLIP->pybel, PLIP will fix the pdb automatically
    try:
        plip_handler = PDBComplex()
        plip_handler.load_pdb(pdb_f, as_string=False)
        mol = plip_handler.protcomplex
        # reading
        l_atoms, r_atoms = get_ligand_rec(mol, ligand_chain, receptor_chain)
        l_atoms, r_atoms, xyz = get_interface_atoms(l_atoms, r_atoms)
    except Exception as e:
        print(f"# [PPIData] error loading <- {pdb_f}")
        return None
    ####
    # Check
    if len(l_atoms) == 0 or len(r_atoms) == 0:
        return None
    # Feature
    if node_featurizer is not None:
        ndata = node_featurizer(l_atoms, r_atoms)
    # Edge featurizer Empty for now
    #####
    # Packing
    ndata = ndata.to(torch.int32)
    xyz = xyz.to(torch.float16)
    lens = [len(l_atoms), len(r_atoms)]
    item = {
        'ndata': ndata,
        'xyz': xyz,
        'lens': lens
    }
    return item
