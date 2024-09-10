import os
import joblib
from tqdm import tqdm


def grep_CCD(f):
    problem_pdbs = ['6a73', '3zjt', '3zju', '3zjv', '6a73', '6myn']
    normal_CCD_list = []
    excludes_pdb_list = []
    for line in f:
        if not line.startswith('#'):
            CCD = line[line.find('(') + 1:-1]
            pdb = line.split()[0]
            CCD = CCD.replace('_', '')
            # add to exclude pdbs list
            if 'mer' in CCD or pdb in problem_pdbs:
                excludes_pdb_list.append([pdb, CCD])
                continue
            # strange tagging, take the first 3 letters
            if len(CCD) > 3:
                CCD = CCD[:3]
                normal_CCD_list.append([pdb, CCD])
            else:
                normal_CCD_list.append([pdb, CCD])
    # record
    # print(normal_CCD_list)
    # print(len(normal_CCD_list))
    return normal_CCD_list, excludes_pdb_list


def display_CCD(ligands_ids):
    for ccd in ligands_ids:
        print(ccd, end=',')

    print(len(ligands_ids))


def download(cmd):
    os.system(cmd)


def download_CCD(CCD):
    normal_CCD_list, excludes_pdb_list = CCD
    # download from online
    cmds = []
    for pdb, ccd in normal_CCD_list:
        output_sdf_f = f'{output_f}/{pdb}_uff.sdf'
        cmd = f'wget https://files.rcsb.org/ligands/view/{ccd}_ideal.sdf -O {output_sdf_f}'
        if not os.path.exists(output_sdf_f):
            cmds.append(cmd)
    joblib.Parallel(n_jobs=20, prefer='threads')(joblib.delayed(download)(x) for x in tqdm(cmds))
    # Copy exclude sdf to designated folder
    # for pdb, ccd in excludes_pdb_list:
    #   os.system(f'cp {ligand_root}/{pdb}_docked.sdf {root}/uff/exclude')


def check():
    import glob
    from rdkit import Chem
    import subprocess
    from rdkit.Chem import rdMolAlign
    gt_ligands = list(glob.glob(f'{ligand_root}/*.sdf'))
    data = {}
    for ligand_f in gt_ligands:
        pdb = os.path.basename(ligand_f)[:4]
        gt_mol = Chem.SDMolSupplier(ligand_f)[0]
        uff_f = f'{root}/uff/final/{pdb}_uff.sdf'
        if gt_mol is None:
            print(pdb, end=',')
        # if os.path.exists(uff_f):
        # print(pdb, end=',')
        # uff_mol = Chem.SDMolSupplier(uff_f)[0]
        # try:
        #   rmsd = rdMolAlign.AlignMol(uff_mol, gt_mol)
        # except Exception as e:
        #   print(pdb, end=',')
        ####
        # cmd = f"obrms {uff_f} {ligand_f}"
        # rmsd_msg = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        # rmsd = float(rmsd_msg.split()[-1].strip())
        # if rmsd == float("inf"):
        #   print(f'{pdb}->{rmsd}', end=',')
        # data[pdb] = rmsd
    return


def download_pdb(CCD):
    normal_CCD_list, excludes_pdb_list = CCD
    pdbs = [x[0] for x in normal_CCD_list] + [x[0] for x in excludes_pdb_list]
    cmds = []
    for pdb in pdbs:
        output_f = f'{root}/pdb'
        cmd = f'wget -q https://files.rcsb.org/view/{pdb}.pdb -O {output_f}/{pdb}.pdb >/dev/null'
        cmds.append(cmd)
    joblib.Parallel(n_jobs=40, prefer='threads')(joblib.delayed(download)(x) for x in tqdm(cmds))


if __name__ == '__main__':
    root = '/opt/home/revoli/data_worker/pdbbind/2020/v2020-other-PL/index'
    ligand_root = '/opt/home/revoli/data_worker/pdbbind/2020/sdf/pdbbank'
    os.makedirs(root + '/pdb', exist_ok=True)
    #
    f1 = open(root + '/INDEX_general_PL_data.2020', 'r').readlines()
    f2 = open(root + '/INDEX_refined_data.2020', 'r').readlines()
    f = f1 + f2
    f = [x.strip() for x in f]
    CCD = grep_CCD(f)
    # check empty row
    special_case = {'5xfj': '6LF', '5xff': '6LF', '3adu': 'MYI'}
    for x in CCD[0]:
        if x[1] == '':
            print(x)
    ########
    download_pdb(CCD)
    ######
    # #
    # CCD_unique = set([x[1] for x in CCD[0]])
    # display_CCD(CCD_unique)
    # print('=' * 100)
    # # download_CCD(CCD)
    # check()
