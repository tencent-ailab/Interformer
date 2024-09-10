from rdkit import Chem
from rdkit.Chem import AllChem
from joblib.externals.loky import set_loky_pickler
from joblib import Parallel, delayed, cpu_count
# from joblib import Parallel, delayed, cpu_count
import warnings
from rdkit import RDLogger
from multiprocessing import Pool

set_loky_pickler('pickle')

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_mol_3d_coordinates(mol):
    try:
        conf = mol.GetConformer()
        conf_num_atoms = conf.GetNumAtoms()
        mol_num_atoms = mol.GetNumAtoms()
        assert mol_num_atoms == conf_num_atoms, \
            'Expect the number of atoms in the molecule and its conformation ' \
            'to be the same, got {:d} and {:d}'.format(mol_num_atoms, conf_num_atoms)
        return conf.GetPositions()
    except:
        warnings.warn('Unable to get conformation of the molecule.')
        return None


def load_molecule(molecule_file, sanitize=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        # prevent some of the pdb, because of valence problem, can't be loaded
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=True)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    if mol:
        if sanitize:
            Chem.SanitizeMol(mol)
        # mol = Chem.RemoveHs(mol)
        pass
    else:
        # None
        print(f"! {molecule_file} can not be read..")

    return mol


def parse_mol(mol):
    if mol is None:
        e = ValueError("Unable to read non None Molecule Object")
        raise e

    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        lig_name = 'LIG'
        # sdf
        if res is None:
            symbol = atom.GetSymbol()
            lig_name = '  LIG' if len(symbol) == 1 else ' LIG'
            atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(' ' + symbol, residueName=lig_name))
            res = atom.GetPDBResidueInfo()
        chain = res.GetChainId()
        ligand_name = res.GetResidueName()
        res.SetResidueName(lig_name)
        res.SetIsHeteroAtom(True)

    return [mol, chain, ligand_name]


def load_sdf(sdf_file, sanitize=False, pose=0):
    try:
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=sanitize, removeHs=True)
    except OSError as e:
        print(f"# unable to load Sdf file {str(sdf_file)}")
        # raise MoleculeLoadException(e)
        return []

    # get pose ligand
    data = {}
    for mol in suppl:
        if mol:
            # mol = Chem.RemoveHs(mol)
            name = mol.GetProp('_Name')
            # if energy > 0, it has clash
            if mol.HasProp('minimizedAffinity'):
                energy = float(mol.GetProp('minimizedAffinity'))
                if energy > 0:
                    # jesus it has clash
                    continue
            if name in data:
                data[name].append(mol)
            else:
                data[name] = [mol]

    # mol processing
    for k, all_mol in data.items():
        multi_poses = []
        # make sure it has enough poses
        if len(all_mol) > 0:
            for i in range(pose):
                # retrieve one pose
                if len(all_mol) >= pose:
                    my_mol = all_mol[i]
                else:
                    my_mol = all_mol[0]
                tmp = parse_mol(my_mol)
                multi_poses.append(tmp)
            # save to dict
            data[k] = multi_poses

    return data


def merge_molecules(ligand, protein):
    """Helper method to merge ligand and protein molecules."""
    from rdkit.Chem import rdmolops
    return rdmolops.CombineMols(ligand, protein)


def pmap(pickleable_fn, data, n_jobs=None, timeout=None, verbose=1, **kwargs):
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    # backend="multiprocessing", prefer='threads'
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=timeout)(
        delayed(pickleable_fn)(d, **kwargs) for i, d in enumerate(data)
    )

    return results


def origin_pmap(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
    # import dill as pickle
    #
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    p = Pool(n_jobs)
    #
    requests = []
    for i in range(len(data)):
        requests.append(p.apply_async(pickleable_fn, args=(data[i],)))
    p.close()
    p.join()
    print('All subprocesses done.')
    results = []
    for res in requests:
        results.append(res.get())
    return results
