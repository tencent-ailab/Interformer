import glob
import os.path
from rdkit import Chem
from Bio.PDB import *
from Bio.PDB.StructureBuilder import StructureBuilder
from scipy.spatial.distance import cdist


def extract_pocket_by_ligand_coord(ligand_coords, ref_ligand_first_id, pdb_mol, output_f, cutoff=10.):
    pdb_atoms = list(pdb_mol.get_atoms())
    pdb_coords = [x.coord for x in pdb_atoms]
    pdb_coords = np.array(pdb_coords)
    # Pocket selection based on cutoff
    dist = cdist(pdb_coords, ligand_coords)
    mask = (dist <= cutoff).any(axis=1)
    mask_idx = np.where(mask)[0]
    #####
    # extend residue
    close_res = set([pdb_atoms[i].parent for i in mask_idx])
    for res in close_res:
        for atom in res:
            atom.saved = True
    ###
    # exclude ref ligand
    ref_ligand = pdb_atoms[ref_ligand_first_id].parent

    class NotDisordered(Select):
        def accept_atom(self, atom):
            close_atom = hasattr(atom, 'saved')
            no_ref = atom.parent != ref_ligand
            not_water = atom.parent.resname != 'HOH'
            return close_atom and not_water and no_ref

    io = PDBIO()
    io.set_structure(pdb_mol)
    io.save(output_f, select=NotDisordered())


def is_equal(c1, c2):
    return (c1 == c2).sum() == 3


def extract_pocket_main(gt_ligand, cutoff=10.):
    pdb = os.path.basename(gt_ligand)[:4]
    pdb_f = f'{pdb_root}/{pdb}.pdb'
    pdb_mol = PDBParser(QUIET=True).get_structure('ag_st', pdb_f)[0]
    sdf_mol = Chem.SDMolSupplier(gt_ligand, sanitize=False)[0]
    ligand_coords = sdf_mol.GetConformer().GetPositions()
    ###
    first_coord = np.array(ligand_coords[0], dtype=np.float32)
    # find out the ligand chain and seq_id, CCD
    ref_ligand_first_id = None
    for i, atom in enumerate(pdb_mol.get_atoms()):
        # handling disorderd
        if atom.is_disordered():
            for key, atom2 in atom.child_dict.items():
                check = is_equal(atom2.coord, first_coord)
                if check:
                    break
        else:  # check
            check = is_equal(atom.coord, first_coord)

        if check:
            ref_ligand_first_id = i
            res = atom.parent
            chain_id = res.parent.id
            is_het = True if res.id[0] != ' ' else False
            if is_het:
                CCD = res.resname
                seq_id = res.id[1]
                cmd = (f'wget -q "https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain_id}&'
                       f'auth_seq_id={seq_id}&encoding=sdf" -O {output_ligand_root}/{pdb}_docked.sdf')
                os.system(cmd)
            else:
                print('pep', pdb, end=',')
                os.system(f'cp {gt_ligand} {output_ligand_root}/{pdb}_docked.sdf')
    ####
    if ref_ligand_first_id is None:
        print('Not matching', pdb)
    # grep pocket
    return 1


if __name__ == '__main__':
    from tqdm import tqdm
    import joblib

    # this script provide a way to find out pocket and download the corresponding ligand from online
    ligand_root = '/opt/home/revoli/data_worker/pdbbind/2020/sdf/pdbbank'
    pdb_root = '/opt/home/revoli/data_worker/pdbbind/2020/v2020-other-PL/index/pdb'
    pocket_root = f'{pdb_root}/pocket'
    output_ligand_root = f'{pdb_root}/ligand'
    os.makedirs(pocket_root, exist_ok=True)
    os.makedirs(output_ligand_root, exist_ok=True)
    #
    gt_ligands = list(glob.glob(f'{ligand_root}/*.sdf'))
    joblib.Parallel(n_jobs=70, prefer='threads')(joblib.delayed(extract_pocket_main)(x) for x in tqdm(gt_ligands))
    # for ligand in gt_ligands:
    #   if '6o0m' in ligand:
    #     extract_pocket_main(ligand)
