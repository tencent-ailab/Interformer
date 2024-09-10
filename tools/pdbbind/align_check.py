import glob
import os.path
from rdkit import Chem
from rdkit.Chem import rdMolAlign

def num_h_atoms(mol):
    n = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            n +=1
    return n


root_1 = '/opt/home/revoli/data_worker/interformer/poses/ligand/rcsb'
pdbbind_root = '/opt/home/revoli/data_worker/pdbbind/2020/sdf/proto'

failed = []
pdbbind_failed = []
download_failed = []
matched = []
rmsd_failed = []
for sdf_f in glob.glob(f'{root_1}/*.sdf'):
    pdb = os.path.basename(sdf_f)[:4]
    m1 = Chem.SDMolSupplier(sdf_f)[0]
    m2 = Chem.SDMolSupplier(f'{pdbbind_root}/{pdb}_docked.sdf')[0]
    if m2 is None:
        # failed pdbbind
        pdbbind_failed.append(pdb)
    if m1 is None:
        download_failed.append(pdb)
    if m1 is None or m2 is None:
        continue
    ###
    try:

        # rmsd = rdMolAlign.AlignMol(m1, m2)
        # if rmsd > 1:
        #     rmsd_failed.append([rmsd, pdb])
        if num_h_atoms(m1) == num_h_atoms(m2):
            matched.append(pdb)
    except Exception as e:
        # print(pdb)
        failed.append(pdb)
        continue
    # print(pdb, rmsd)

# print(download_failed, len(download_failed))
# print(failed, len(failed))
# print(pdbbind_failed, len(pdbbind_failed))
# print(rmsd_failed)

# print(pdbbind_failed)

selected = matched
f = open('/opt/home/revoli/data_worker/pdbbind/2020/sdf/match', 'w')
f.write('\n'.join(matched))
f.close()
full = [os.path.basename(x)[:4] for x in glob.glob(pdbbind_root + '/*.sdf')]
pdbbind = list(set(full) - set(matched))
f = open('/opt/home/revoli/data_worker/pdbbind/2020/sdf/pdbbind', 'w')
f.write('\n'.join(pdbbind))
f.close()
