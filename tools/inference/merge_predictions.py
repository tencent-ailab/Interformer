import pandas as pd


if __name__ == '__main__':
    test_csvs = ['lsd1_project_ensemble', 'mPro_covalent_test_ensemble', 'mPro_project_ensemble']
    for test_csv in test_csvs:
        df = pd.read_csv(f'/opt/home/revoli/eva/Interformer/result/neg_affinity/{test_csv}.csv')
        normal = pd.read_csv(f'/opt/home/revoli/eva/Interformer/result/normal/{test_csv}.csv')
        normal = normal[['Target', 'Molecule ID', 'pred_pIC50']]
        normal = normal.rename(columns={'pred_pIC50': 'normal_affinity'})
        pgnn = pd.read_csv(f'/opt/home/revoli/eva/Interformer/result/pGNN/{test_csv}.csv')
        pgnn = pgnn[['Target', 'Molecule ID', 'pred_pIC50']]
        pgnn = pgnn.rename(columns={'pred_pIC50': 'pGNN'})
        df = df.merge(normal, on=['Target', 'Molecule ID'])
        df = df.merge(pgnn, on=['Target', 'Molecule ID'])
        df.to_csv(f'/opt/home/revoli/eva/Interformer/result/neg_affinity/merge/{test_csv}.csv', index=False)
