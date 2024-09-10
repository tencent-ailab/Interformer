# This script for evaluation of the results of interformer
# the performance not good
import pandas as pd


def eval(df, name):
    corr = df.corr(numeric_only=True)
    print(name)
    print(corr['pIC50'])
    print('=' * 100)


if __name__ == '__main__':
    df = pd.read_csv('/opt/home/revoli/eva/Interformer/result/6w4k_inter_infer_ensemble.csv')
    df = df.drop(columns=['pIC50', 'uff_pose_rank'])
    gt = pd.read_csv('/opt/home/revoli/data_worker/paper/NC/benchmark/affinity/lsd1_project.csv')
    gt = gt.drop(columns=['pred_pose', 'pred_pIC50', 'pose_rank'])
    final = df.merge(gt, on='Molecule ID')
    # Grep Best PoseScore
    test = final[final['inter_rank'] != 20]  # exclude ref_ligands
    top1 = test.sort_values('pred_pose', ascending=False).groupby('Molecule ID').head(1)
    eval(top1, 'Top1 W/O ref')
    ##
    # Grep Ref+Best
    top1 = final.sort_values('pred_pose', ascending=False).groupby('Molecule ID').head(1)
    eval(top1, 'Top1 W/ ref')
    ###
    ref_df = final[final['inter_rank'] == 20]
    eval(ref_df, 'Ref')
