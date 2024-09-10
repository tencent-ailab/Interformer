import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plip.structure.detection
from plip.structure.preparation import PDBComplex
from rdkit import Chem
from rdkit.Geometry import Point3D
from collections import defaultdict
from tqdm import tqdm


def plip_load(complex_str):
    plip_handler = PDBComplex()
    plip_handler.load_pdb(complex_str, as_string=True)
    for ligand in plip_handler.ligands:
        plip_handler.characterize_complex(ligand)
    itypes = plip_handler.interaction_sets['UNL:Z:1'].all_itypes
    res = defaultdict(list)
    for itype in itypes:
        if len(itype) == 19:  # hbond
            res['hbond'].append([itype.a_orig_idx, itype.d_orig_idx])
        if len(itype) == 11:  # hydrophobic
            res['hydro'].append([itype.bsatom_orig_idx, itype.ligatom_orig_idx])
    for key in res:
        res[key].sort(key=lambda x: x)
    return res

def rm_UNL_H(complex_str):
    data = []
    ligand_num = 0
    for line in complex_str.split('\n'):
        if 'UNL' in line:
            atomic = line[77]
            if atomic != 'H':
                data.append(line)
                ligand_num += 1
        else:
            data.append(line)
    data = '\n'.join(data)
    return data, ligand_num

# extract pocket from complex.pdb, and merge with sdf
def merge_complex_with_sdf(sdf_f, complex_f, sdf_id=0):
    pdb_mol = Chem.MolFromPDBFile(complex_f, sanitize=False, removeHs=False)
    ligand_mol = list(Chem.SDMolSupplier(sdf_f, sanitize=False, removeHs=True))[sdf_id]
    ligand_mol = Chem.RemoveHs(ligand_mol)
    # Merge
    complex = Chem.rdmolops.CombineMols(ligand_mol, pdb_mol)
    complex_str = Chem.MolToPDBBlock(complex)
    complex_str, ligand_num = rm_UNL_H(complex_str)
    # remove header
    if complex_str.startswith('COMP'):
        complex_str = complex_str[complex_str.find('\n') + 1:]
    return complex_str, ligand_num


def eval(docked_data, crystal_data):
    # compare correct
    data = []
    for key in crystal_data:
        correct = 0
        for inter in crystal_data[key]:
            if key in docked_data:
                for inter2 in docked_data[key]:
                    if inter == inter2:
                        correct += 1
        # print(f"{key}->{correct}/{len(crystal_data[key])}")
        N = len(crystal_data[key])
        data.append([key, correct, N, correct / N])
    return data


def eval_hit_rate(df, name):
    print(name)
    print(f"n:{len(df['pdb'].unique().tolist())}")
    print(f"Hbond-Mean_hit_rate:{df[df['inter_type'] == 'Hbond']['hit_rate'].mean()}")
    print(f"Hydro-Mean_hit_rate:{df[df['inter_type'] == 'Hydrophobic']['hit_rate'].mean()}")
    print('-' * 100)


if __name__ == '__main__':
    pre = 0
    use_pose_score = 0
    method_name = 'Interformer'  # Interformer, DiffDock, DeepDock
    output_folder = '/opt/home/revoli/data_worker/paper/benchmark/docking/pose_stats'
    output_name = method_name
    if method_name == 'Interformer':
        if use_pose_score:
            output_name += '-PoseScore'
        else:
            output_name += '-Energy'
    tmp_f = f'{output_folder}/{output_name}_pose2stats.csv'
    print(tmp_f)
    ###
    test_list = [x.strip() for x in
                 open('/opt/home/revoli/data_worker/interformer/train/diffdock_splits/timesplit_test').readlines()]
    root = '/opt/home/revoli/eva/Interformer/dock_results/energy_timetest'
    # test_list = ['6hld']
    ###########
    if pre:
        data = []
        sdf_id = 0
        ref_id = 0
        # load ranking data
        if method_name == 'Interformer':
            pose_sel_df = pd.read_csv('/opt/home/revoli/eva/Interformer/result/core_timetest.round0_ensemble.csv')
        #
        for t in tqdm(test_list):
            complex_f = f'{root}/complex/{t}_complex.pdb'
            gt_sdf_f = f'{root}/ligand/{t}_docked.sdf'
            if not os.path.exists(complex_f) and not os.path.exists(gt_sdf_f):
                continue
            # docked_sdf
            if method_name == 'Interformer':
                sdf_f = f'{root}/ligand_reconstructing/{t}_docked.sdf'
                sdf_id = 0
                if use_pose_score:
                    target_df = pose_sel_df[pose_sel_df['Target'] == t]
                    if len(target_df):
                        # exclude crystal and get the maximum pose_rank's pose_id
                        sdf_id = int(target_df.head(20).sort_values('pred_pose', ascending=False).head(1)['pose_rank'])
                    else:
                        sdf_id = 0
                else:
                    sdf_id = 0

            elif method_name == 'DiffDock':
                sdf_f = f'/opt/home/revoli/git/DiffDock/results/user_predictions_small/{t}/rank1.sdf'  # diffdock
            elif method_name == 'DeepDock':
                sdf_f = f'/opt/home/revoli/git/DeepDock/output/{t}_docked.sdf'
            # statistic
            try:
                print(t, end=',')
                docked_str, dock_num = merge_complex_with_sdf(sdf_f, complex_f, sdf_id=sdf_id)
                gt_str, gt_num = merge_complex_with_sdf(gt_sdf_f, complex_f, sdf_id=ref_id)
                assert dock_num == gt_num
                docked_data = plip_load(docked_str)
                crystal_data = plip_load(gt_str)
            except Exception as e:
                print(f"PLIP error->{t}, {e}")
                continue
            # compare & eval
            eval_data = eval(docked_data, crystal_data)
            eval_data = [[t] + x for x in eval_data]
            data.extend(eval_data)
        df = pd.DataFrame(data, columns=['pdb', 'inter_type', 'correct', 'num_inter', 'hit_rate'])
        df.to_csv(tmp_f, index=False)
        print(df)
    #######
    ## load
    methods = ['DiffDock', 'DeepDock', 'Interformer-PoseScore', 'Interformer-Energy']
    all_df = []
    for method in methods:
        a = pd.read_csv(f'{output_folder}/{method}_pose2stats.csv')  # tmp test, interformer_pose2stats
        a['method'] = method
        all_df.append(a)
    df = pd.concat(all_df).reset_index(drop=True)
    df['inter_type'] = df['inter_type'].map(lambda x: x.capitalize())
    df['inter_type'] = df['inter_type'].map(lambda x: x if x == 'Hbond' else 'Hydrophobic')
    df['Interactions'] = df['method'] + '-' + df['inter_type']
    # scale hit rate 100 times
    df['hit_rate'] *= 100.
    ######
    # Mean
    for method in methods:
        eval_hit_rate(df[df['method'] == method], name=method)
    #####
    # confirmed pdbs
    # new_df = df[df['method'] == 'Interformer']
    # w = open('/opt/home/revoli/data_worker/interformer/train/diffdock_splits/timesplit_test_sanitizable', 'w')
    # w.write('\n'.join(new_df['pdb'].unique().tolist()))
    # w.close()
    ###
    print("DONE")
