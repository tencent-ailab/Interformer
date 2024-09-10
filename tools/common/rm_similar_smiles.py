import pandas as pd

if __name__ == '__main__':
    print("Remove bad targets and overlapping smiles to training set")
    df = pd.read_csv('/opt/home/revoli/data_worker/paper/benchmark/docking/similarity/kinase_test.csv')
    df = df[df['simiarity'] > 0.7]
    smiles = df['Smiles']
    # origin input csv
    ori_csv_f = '/opt/home/revoli/data_worker/paper/benchmark/affinity/kinase_test.csv'
    df = pd.read_csv(ori_csv_f)
    print(len(df))
    good_pdbs = ['1h00', '830c', '3lpb', '2rgp', '2ojg', '2i78', '2zec', '2zdt', '3eqh', '2vt4', '3bz3', '3chp', '2hzi',
                 '1sqt', '2oj9', '3hmm', '3c4f', '2p2i', '3eml', '3m2w', '2ayw', '3g6z', '3pbl', '3d4q', '3el8', '3biz',
                 '2qd9', '2etr', '3krj', '3l5d', '2oi0', '2owb', '3odu', '2of2', '1ype', '3g0e', '3ny8', '3lq8', '3bkl',
                 '3kl6']
    df = df[df['Target'].isin(good_pdbs)]
    df = df[~df['Smiles'].isin(smiles)]
    # output
    df.to_csv(ori_csv_f, index=False)
    print(len(df))
    pass