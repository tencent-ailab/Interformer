# this script calculate the accuracy of diffdock methods, finally we will have a summary csv contains
# the results of diffdock
import glob
import os
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm


def compute_rmsd(query, target_pdb):
    gt_root = '/opt/home/revoli/data_worker/interformer/poses/ligand/pdbbank'
    cmd = f"obrms {query} {gt_root}/{target_pdb}_docked.sdf"
    rmsd_msg = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    rmsd = float(rmsd_msg.split()[-1])
    return rmsd


if __name__ == "__main__":
    df = pd.read_csv('/opt/home/revoli/git/DiffDock/data/testset_csv.csv')
    result_f = '/opt/home/revoli/git/DiffDock/results/user_predictions_testset'
    os.makedirs(f'{result_f}/output', exist_ok=True)
    #
    data = []
    for i, row in tqdm(df.iterrows()):
        pdb = row['protein_path'].split('/')[2]
        for j in range(1, 41):
            src_files = list(glob.glob(f'{result_f}/{i}/rank{j}_con*.sdf'))
            if len(src_files):
                query_f = src_files[0]
                confident_score = query_f.split('/')[-1].replace('-', '')
                confident_score = float(confident_score[confident_score.find('dence') + 5:-4])
                rmsd = compute_rmsd(query_f, pdb)
                data.append([pdb, j, confident_score, rmsd])
            else:
                print(pdb, i, j)  # error

    df = pd.DataFrame(data, columns=['Target', 'pose_rank', 'confidence', 'rmsd'])
    df.to_csv(f'{result_f}/output/diff_summary.csv', index=False)
    print("DONE")
