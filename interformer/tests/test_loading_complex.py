# it is a test for using rdkit to read complex and extract pocket by assigning a ligand
# a simple tested written by revo
# To see how to maintain ligand in the pocket.
from rdkit import Chem
from oddt.toolkits.extras.rdkit import fixer


def create_ligand_residue(complex):
    for atom in complex.GetAtoms():
        res = atom.GetPDBResidueInfo()
        # sdf
        if res is None:
            symbol = atom.GetSymbol()
            lig_name = '  LIG' if len(symbol) == 1 else ' LIG'
            atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(' ' + symbol, residueName=lig_name))
            res = atom.GetPDBResidueInfo()
            res.SetResidueName(lig_name)
            res.SetIsHeteroAtom(True)
        else:
            if res.GetIsHeteroAtom():
                res.SetIsHeteroAtom(False)
    # debug
    # str_ = Chem.MolToPDBBlock(complex)
    pass


def print_res_info(atoms):
    for atom in atoms:
        res = atom.GetPDBResidueInfo()
        print(res.GetResidueName())


if __name__ == '__main__':
    append_residues = ['ZN', 'CL', 'MG']
    root = '/opt/home/revoli/data_worker/interformer/test/posebuster/inter_posebuster'
    pdb = '8H0M'
    pdb_mol = Chem.MolFromPDBFile(f'{root}/pocket/{pdb}_pocket.pdb', sanitize=False, removeHs=True)
    ligand = Chem.SDMolSupplier(f'{root}/ligand/{pdb}_docked.sdf')[0]
    complex = Chem.rdmolops.CombineMols(ligand, pdb_mol)
    create_ligand_residue(complex)
    pdb_atoms = pdb_mol.GetAtoms()
    c_atoms = complex.GetAtoms()
    pocket, ligand = fixer.ExtractPocketAndLigand(complex, cutoff=10, append_residues=append_residues,
                                                  expandResidues=True)
    p_atoms, l_atoms = pocket.GetAtoms(), ligand.GetAtoms()
    print_res_info(p_atoms)
    print('done')
