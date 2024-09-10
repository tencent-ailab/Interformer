# This script trys to generate correct UFF conformation of ligands from sdf
# a reference script for generating correct uff conformation
# the molecule must be sanitized.
import glob
import os.path
import sys
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs, ETKDGv3
from copy import deepcopy
from rdkit.Chem.rdForceFieldHelpers import (
    UFFGetMoleculeForceField,
    UFFOptimizeMoleculeConfs,
)
from tqdm import tqdm
import joblib
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import AddHs, AssignStereochemistryFrom3D
from rdkit.Chem import rdMolAlign


def calculate_rmsd(mol1, mol2):
    rmsd = rdMolAlign.AlignMol(mol1, mol2)
    return rmsd


def new_conformation(mol: Mol, n_confs: int = 30, num_threads: int = 0, energy_minimization=True) -> Mol:
    """Generate new conformation(s) for a molecule."""
    if mol is None:
        return None

    etkdg = ETKDGv3()
    etkdg.randomSeed = 42
    etkdg.verbose = False
    etkdg.useRandomCoords = True
    etkdg.numThreads = num_threads

    try:
        # prep mol
        mol_etkdg = deepcopy(mol)
        mol_etkdg = AddHs(mol_etkdg, addCoords=True)
        AssignStereochemistryFrom3D(mol_etkdg, replaceExistingTags=True)
        mol_etkdg.RemoveAllConformers()
        cids_etkdg = EmbedMultipleConfs(mol_etkdg, n_confs, etkdg)
    except Exception as e:
        print(e)
        return None
    ####
    if len(cids_etkdg) == 0:
        print('Failed to generate conformations.', cids_etkdg, len(cids_etkdg))
        return None

    if energy_minimization:
        min_E = 9999999.
        min_confId = None
        try:
            data = UFFOptimizeMoleculeConfs(mol_etkdg, numThreads=20, maxIters=10000)
        except Exception as e:
            print('failed in uff optimization', e)
            return None
        energies = [v[1] for v in data]
        min_E = min(energies)
        min_confId = energies.index(min_E)
        # get the best
        minEConfMol = Chem.Mol(mol_etkdg, False, min_confId)
        rmsd = calculate_rmsd(mol, minEConfMol)
        minEConfMol.SetProp('uff_energy', str(min_E))
        return {"mol": minEConfMol, "energy": min_E, 'rmsd': rmsd}


def run_fn(sdf_f):
    pdb = os.path.basename(sdf_f)[:4]
    output_f = f'{sys.argv[2]}/{pdb}_uff.sdf'
    if os.path.exists(output_f):
        return 0
    #
    supplier = Chem.SDMolSupplier(sdf_f, sanitize=True)
    mol = next(supplier)
    result = new_conformation(mol)
    if result is None:
        print(sdf_f)
        os.system(f'cp {sdf_f} {output_f}')
        return 1
    print(f"{pdb}, RMSD:{result['rmsd']}, E:{result['energy']}")
    # output
    writer = Chem.SDWriter(output_f)
    writer.write(result['mol'])
    writer.close()
    return 0


if __name__ == "__main__":
    input_root = sys.argv[1]
    output_root = sys.argv[2]
    # read molecule from sdf files
    sdfs = list(glob.glob(input_root + '/*.sdf'))
    # print(sdfs)
    joblib.Parallel(n_jobs=70, prefer='threads')(joblib.delayed(run_fn)(x) for x in tqdm(sdfs))
