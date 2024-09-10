# This is a script for downloading pdb from rcsb by pdb_list
# we regenerate pocket from rcsb raw data
import os
import joblib
from tqdm import tqdm


def download_pdb(items):
    pdb, output_f = items
    wget_cmd = f'wget -q https://files.rcsb.org/download/{pdb}.pdb -O {output_f}'
    # print(wget_cmd)
    os.system(wget_cmd)


if __name__ == '__main__':
    output_path = '/opt/home/revoli/data_worker/interformer/rebuild_pocket/pdb'
    os.makedirs(output_path, exist_ok=True)
    query = [(x.strip(), f'{output_path}/{x.strip()}.pdb') for x in
             open('/opt/home/revoli/data_worker/interformer/rebuild_pocket/pdb_list').readlines()]
    # we run it in Parallel by using joblib, it is a easy way to go
    joblib.Parallel(n_jobs=70, prefer='threads')(joblib.delayed(download_pdb)(x) for x in tqdm(query))
