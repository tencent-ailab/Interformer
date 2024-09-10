# This is a simple demo to call the python API of posebuster
# import posebusters
from posebusters import PoseBusters
from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from rdkit import Chem

if __name__ == '__main__':
    root = '/opt/home/revoli/eva/Interformer/energy_output/ligand_reconstructing'
    pdb = '5SAK'
    pdb_mol = Chem.MolFromPDBFile(f'{root}/../pocket/{pdb}_pocket.pdb')
    sdf_mol = Chem.SDMolSupplier(f'{root}/{pdb}_docked.sdf')[0]
    fn = check_intermolecular_distance(sdf_mol, pdb_mol)
    print('done')
