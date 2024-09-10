import glob
import itertools
import os
import subprocess
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm


def cal_bust(sdf_f):
    pdb = os.path.basename(sdf_f)[:4]
    pocket, ref = f'{pocket_path}/{pdb}_pocket.pdb', f'{base_folder}/ligand/{pdb}_docked.sdf'
    cmd = f'bust {sdf_f} -l {ref} -p {pocket}  --outfmt csv  --top-n 1'
    result_msg = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    result = [x.split(',') for x in result_msg.split('\n')[1:-1]]
    # filter and add pose_rank
    result = [[i, pdb] + x[1:] for i, x in enumerate(result)]
    return result


def eval(df, check_lists, name='full'):
    print(name)
    print(f"AC:{df['rmsd_≤_2å'].sum()}/{len(df)}->{round(df['rmsd_≤_2å'].sum() * 100. / len(df), 4)}")
    df['pass'] = df[check_lists].sum(axis=1) == len(check_lists)
    df['score'] = df[check_lists].sum(axis=1)
    pass_and_rmsd_ok = df[df['pass'] == True]
    print(
        f"PASS & AC:{pass_and_rmsd_ok['rmsd_≤_2å'].sum()}/{len(df)}"
        f"->{round(pass_and_rmsd_ok['rmsd_≤_2å'].sum() * 100. / len(df), 4)}")
    print(f'Pass:{df["pass"].sum()}/{len(df)}->{round(df["pass"].sum() * 100. / len(df), 4)}')
    print('=' * 100)


if __name__ == '__main__':
    # folder = ['ablation/nomask', 'ablation/noedge', 'ablation/nointer', 'ablation/nointra', 'energy_output']
    # folder = ['dock_results/energy_timetest']
    ###
    # Posebuster
    base = '/opt/home/revoli/eva/Interformer'
    folder = ['dock_results/energy_timetest']  # energy_posebuster
    ligand_folder = 'noVdw'  # ref, ligand_reconstructing, posescore
    ##
    for f in folder:
        base_folder = f'/{base}/{f}'
        ligand_root = f'{base_folder}/{ligand_folder}'
        pocket_path = f'{base_folder}/pocket'
        ori_check_lists = ["mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded", "sanitization",
                           "all_atoms_connected",
                           "molecular_formula", "molecular_bonds", "double_bond_stereochemistry",
                           "tetrahedral_chirality", "bond_lengths",
                           "bond_angles", "internal_steric_clash", "aromatic_ring_flatness", "double_bond_flatness",
                           "internal_energy",
                           "protein-ligand_maximum_distance", "minimum_distance_to_protein",
                           "minimum_distance_to_organic_cofactors",
                           "minimum_distance_to_inorganic_cofactors", "minimum_distance_to_waters",
                           "volume_overlap_with_protein",
                           "volume_overlap_with_organic_cofactors", "volume_overlap_with_inorganic_cofactors",
                           "volume_overlap_with_waters"]
        exclude_lists = ["minimum_distance_to_waters", "volume_overlap_with_waters"]
        check_lists = list(set(ori_check_lists) - set(exclude_lists))
        #
        paper_check_lists = ["docked_ligand_successfully_loaded", "molecule_passes_rdkit_sanity_check",
                             "molecular_formula_preserved",
                             "molecular_bonds_preserved", "sp3_stereochemistry_preserved",
                             "double_bond_stereochemistry_preserved",
                             "bond_lengths_within_bounds",
                             "bond_angles_within_bounds", "no_internal_clashes", "aromatic_ring_flatness_passes",
                             "double_bond_flatness_passes",
                             "energy_ratio_within_threshold", "no_clashes_with_protein",
                             "no_clashes_with_organic_cofactors",
                             "no_clashes_with_inorganic_cofactors", "no_volume_clash_with_protein",
                             "no_volume_clash_with_organic_cofactors",
                             "no_volume_clash_with_inorganic_cofactors"]
        sdfs = list(glob.glob(f'{ligand_root}/*.sdf'))
        # #####
        # # Process
        data = joblib.Parallel(n_jobs=70, prefer='threads')(joblib.delayed(cal_bust)(x) for x in tqdm(sdfs))
        data = list(itertools.chain.from_iterable(data))
        df = pd.DataFrame(data, columns=["pose_rank", "pdb", "molecule"] + ori_check_lists +
                                        ["rmsd_≤_2å"])
        # energy
        df = df[df['pose_rank'] == 0]  # only save the best result
        df.to_csv('/tmp/posebuster_check.csv', index=False)
        ###
        # Basic Info
        df = pd.read_csv('/tmp/posebuster_check.csv')
        eval(df, check_lists)
        v2_pdbs = [x[:4] for x in open(
            '/opt/home/revoli/data_worker/interformer/train/diffdock_splits/posebusters_pdb_ccd_ids.txt').readlines()]
        v2_df = df[df['pdb'].isin(v2_pdbs)].copy()
        eval(v2_df, check_lists, name='v2')
        #
    print('done')
