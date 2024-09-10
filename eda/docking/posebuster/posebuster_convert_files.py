import glob
import os
import shutil
import pandas as pd

if __name__ == '__main__':
    root = '/opt/home/revoli/data_worker/benchmark/posebuster'
    output_path = f'{root}/inter_posebuster'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '/ligand', exist_ok=True)
    os.makedirs(output_path + '/pocket', exist_ok=True)
    os.makedirs(output_path + '/uff', exist_ok=True)
    csv_data = []
    for folder in glob.glob(f'{root}/posebusters_benchmark_set/*'):
        pdb_ccd = os.path.basename(folder)
        pdb = pdb_ccd[:4]
        shutil.copy(f'{folder}/{pdb_ccd}_ligand.sdf', f'{output_path}/ligand/{pdb}_docked.sdf')  # copy gt ligand
        shutil.copy(f'{folder}/{pdb_ccd}_ligand_start_conf.sdf',
                    f'{output_path}/uff/{pdb}_uff.sdf')  # copy start ligand
        shutil.copy(f'{folder}/{pdb_ccd}_protein.pdb', f'{output_path}/pocket/{pdb}_pocket.pdb')

        m_id = open(f'{folder}/{pdb_ccd}_ligand.sdf').readlines()[0].strip()
        csv_data.append([pdb, 0., m_id])
    #
    df = pd.DataFrame(data=csv_data, columns=['Target', 'pIC50', 'Molecule ID'])
    df.to_csv(f'{output_path}/posebuster_infer.csv', index=False)
    print('done')
