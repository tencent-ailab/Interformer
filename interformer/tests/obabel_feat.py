from rdkit import Chem
from openbabel import openbabel, pybel

from feats.gnina_types.gnina_featurizer import PLIPAtomFeaturizer


def grep_mol():
    sdf_f = '/opt/home/revoli/data_worker/project/cfDNA/uff/7kiu_uff.sdf'
    mol = Chem.SDMolSupplier(sdf_f)[0]
    mol_str = Chem.MolToMolBlock(mol)
    mol = pybel.readstring('sdf', mol_str)
    # all_mols = pybel.readfile('sdf', sdf_f)
    # mol = next(all_mols)
    return mol


if __name__ == '__main__':
    mol = grep_mol()
    mol2 = grep_mol()
    featurizer = PLIPAtomFeaturizer()
    feat = featurizer(mol2, mol)
    print('done')
