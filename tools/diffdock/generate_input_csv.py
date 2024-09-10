import pandas as pd

if __name__ == '__main__':
    # this script is genreated for diffdock as input
    root = '/opt/home/revoli/data_worker/paper/benchmark/docking/core_timetest.csv'
    targets = pd.read_csv(root)['Target'].unique().tolist()
    data = [[t, f'docking/ligand/{t}_docked.sdf', f'docking/protein/{t}.pdb', ''] for t in targets]
    data = pd.DataFrame(data=data, columns=['complex_name', 'ligand_description', 'protein_path', 'protein_sequence'])
    data.to_csv('/opt/home/revoli/git/DiffDock/core_test.csv', index=False)
