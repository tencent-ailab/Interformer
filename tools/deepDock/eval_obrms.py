import glob
import os
import subprocess
from tqdm import tqdm

import pandas as pd

if __name__ == '__main__':
    # root = '/opt/home/revoli/git/DeepDock/output'
    root = '/opt/home/revoli/git/DiffDock/results/user_predictions_small/gather'
    gt_root = '/opt/home/revoli/data_worker/interformer/poses/ligand/final'
    sdf_files = list(glob.glob(f'{root}/*.sdf'))
    data = []
    for sdf_f in tqdm(sdf_files):
        base_name = os.path.basename(sdf_f)
        gt_f = f'{gt_root}/{base_name}'
        pdb = base_name[:4]
        cmd = f'obrms {sdf_f} {gt_f}'
        result_msg = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        rmsd = float(result_msg.split()[-1])
        data.append([pdb, 0, rmsd])

    df = pd.DataFrame(data=data, columns=['Target', 'pose_rank', 'rmsd'])
    df.to_csv(f'{root}/obrms_rmsd.csv', index=False)
