import os.path

from rdkit import Chem
from collections import defaultdict
from rdkit import RDLogger
from rdkit.Geometry import Point3D
import torch
from tqdm import tqdm

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def sdf_load_full(sdf_f):
    try:
        suppl = list(Chem.SDMolSupplier(sdf_f, sanitize=True, removeHs=True))
        return suppl
    except OSError as e:
        print(f"# error load<-{sdf_f}")
        return None


def sdf_load(row, use_mid=False, warning=True):
    sdf_file, mid, id, pdb = row  # id_name may be pose_rank or Molecule ID
    try:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=True)
    except OSError as e:
        if warning:
            print(f"# error load<-{sdf_file}")
        return None
    ligands = list(suppl)
    # retrieve specific mol
    if use_mid:
        matched_id_ligands = []
        for l in ligands:
            if l:
                name = l.GetProp('_Name')
                if name == mid:
                    matched_id_ligands.append(l)
        ligands = matched_id_ligands
    # using pose_rank to select
    if id < len(ligands):
        id_name = f'{pdb}_{mid}_{str(id)}'
        return ligands[id], id_name

    return None  # failed

def load_sdf_at_once(sdfs_data, use_mid=False):
    print(f"# [vs mode] loading sdf_file at once")
    # sdfs_data=[sdf_file, mid, id, pdb]
    unique_sdf_file = set([x[0] for x in sdfs_data])
    sdf_f_to_mol = {}
    for sdf_file in unique_sdf_file:
        try:
            suppl = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=True)
            sdf_f_to_mol[sdf_file] = suppl
        except OSError as e:
            print(f"# error load<-{sdf_file}")
            return None
    # 2. get the specific ligand from sdf
    data = []
    for row in tqdm(sdfs_data):
        sdf_file, mid, id, pdb = row
        # extract ligand from mol
        ligands = sdf_f_to_mol[sdf_file]
        if use_mid:
            matched_id_ligands = []
            for l in ligands:
                if l:
                    name = l.GetProp('_Name')
                    if name == mid:
                        matched_id_ligands.append(l)
            ligands = matched_id_ligands
        if id < len(ligands):
            id_name = f'{pdb}_{mid}_{str(id)}'
            data.append([ligands[id], id_name])
        else:
            data.append(None)
    #
    return data


def load_by_rdkit(input_file, format='sdf', pose=0, warning=True):
    if format == 'sdf':
        try:
            suppl = Chem.SDMolSupplier(input_file, sanitize=True, removeHs=True)
        except OSError as e:
            if warning:
                print(f"# unable to load Sdf file {input_file}")
            return None
        # get pose ligand
        data = defaultdict(list)
        for mol in suppl:
            if mol:
                name = mol.GetProp('_Name')
                # if energy > 0, it has clash
                if mol.HasProp('minimizedAffinity'):
                    energy = float(mol.GetProp('minimizedAffinity'))
                    if energy > 0:
                        # jesus it has clash
                        continue
                data[name].append(mol)
        # mol processing
        for k, all_mol in data.items():
            multi_poses = []
            # make sure it has enough poses
            if len(all_mol) > 0:
                for i in range(pose + 1):
                    # retrieve one pose
                    if len(all_mol) > i:
                        my_mol = all_mol[i]
                    else:
                        my_mol = all_mol[0]
                    multi_poses.append(my_mol)
                # save to dict
                data[k] = multi_poses
        return data
    elif format == 'pdb':
        mol = Chem.MolFromPDBFile(input_file, sanitize=False, removeHs=True)
        mol = Chem.RemoveHs(mol, sanitize=False)
        return mol


def rdkit_write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    if isinstance(new_coords, torch.Tensor):
        new_coords = new_coords.numpy()
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        x, y, z = new_coords[i].tolist()
        conf.SetAtomPosition(i, Point3D(x, y, z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()
