import glob
import os.path
from tqdm import tqdm
import sys
import joblib


def run_fn(pdb_f):
    base_name = os.path.basename(pdb_f)
    os.system(f'reduce -Quiet {pdb_f} > {output_folder}/{base_name}')


if __name__ == '__main__':
    root = sys.argv[1]
    pdbs_f = glob.glob(f'{root}/*.pdb')
    output_folder = f'{root}/reduce'
    os.makedirs(output_folder, exist_ok=True)
    joblib.Parallel(n_jobs=70, prefer='threads')(joblib.delayed(run_fn)(x) for x in tqdm(pdbs_f))
