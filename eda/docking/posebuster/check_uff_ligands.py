# It is for checking can uff be aligned to ref ligand, by AlignMol of Rdkit
# It is for checking can uff be aligned to ref ligand, by AlignMol of Rdkit
import glob, os

from rdkit import Chem
from rdkit.Chem import rdMolAlign


def align(uff_f, ligand_f):
    mol1 = Chem.SDMolSupplier(uff_f)[0]
    mol2 = Chem.SDMolSupplier(ligand_f)[0]
    try:
        rmsd = rdMolAlign.AlignMol(mol1, mol2)
    except Exception as e:
        # print('error:', uff_f)
        return -1.
    return rmsd


if __name__ == "__main__":
    uff_root = '/opt/home/revoli/eva/Interformer/energy_output/uff'
    ligand_root = '/opt/home/revoli/eva/Interformer/energy_output/ligand'
    e = 0
    for uff_f in glob.glob(f'{uff_root}/*.sdf'):
        pdb = os.path.basename(uff_f)[:4]
        ligand_f = f'{ligand_root}/{pdb}_docked.sdf'
        rmsd = align(uff_f, ligand_f)
        if rmsd == -1. or rmsd > 3.:
            print(pdb, rmsd)
            e += 1
    print(e)  # 5 failed, 59 greater than RMSD=3
