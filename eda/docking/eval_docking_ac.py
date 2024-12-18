import pandas as pd
import sys
from sklearn import metrics


def eval(df, name, top=1):
    top1 = df.groupby(target_key).head(top)
    top1 = top1.groupby(target_key).min()
    hit = (top1['rmsd'] < 2.).sum()
    N = len(top1)
    print(f"N:{N}")
    print(name, f'{hit} / {N} ->', hit * 100. / N)
    print(name + '_median', top1['rmsd'].median())


def main(df, name):
    print(name)
    if len(df) == 0:
        print("testing dataframe len=0")
        return
    # Pose-Sel, top1
    if 'pred_pose' in df:
        print("Using PoseScore for ranking.")
        pred_df = df.sort_values('pred_pose', ascending=0)
        eval(pred_df, "PoseSel-top1")
        pred_df = df.sort_values('pred_pose', ascending=0)
        eval(pred_df, "PoseSel-top5", top=5)
        pred_df = df.sort_values('pred_pose', ascending=0)
        eval(pred_df, "PoseSel-top10", top=10)
        print("%" * 50)
        # eval AUROC
        fpr, tpr, thresholds = metrics.roc_curve(df['rmsd'] < 2., df['pred_pose'], pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        print(f"PredPose/RMSD->[AUROC={roc_auc}]")
        print("=" * 50)
    # Rank by energy
    if 'pose_rank' in df:
        print("Using Energy for ranking.")
        pred_df = df.sort_values('pose_rank', ascending=1)
        eval(pred_df, "PoseRank-top1")
        pred_df = df.sort_values('pose_rank', ascending=1)
        eval(pred_df, "PoseRank-top5", top=5)
        pred_df = df.sort_values('pose_rank', ascending=1)
        eval(pred_df, "PoseRank-top10", top=10)
    print("*" * 50)
    print("*" * 50)


if __name__ == '__main__':
    print('[input_csv] [poserank2exclude]')
    # INPUT
    df = pd.read_csv(sys.argv[1])
    if len(sys.argv) > 2:
        pose_rank_exclude = int(sys.argv[2])
    else:
        pose_rank_exclude = 20
    #
    target_key = 'Target' if 'Target' in df else 'pdb_id'
    df = df[df['pose_rank'] != pose_rank_exclude]  # exclude crystal
    print(f"Excluding pose_rank = {pose_rank_exclude} (crystal structure)")
    #
    main(df, name='Full')
    ###
    time_split_pdbs = set([x.strip() for x in open(
        'eda/docking/splits/timesplit_test_sanitizable').readlines()])
    time_df = df[df[target_key].isin(time_split_pdbs)]
    if len(time_df):
        # print(f"Exclude: {exlclude_pdbs}")
        main(time_df, name="Time Split Test")
    ###
    posebuster_v2_pdbs = set([x.strip()[:4] for x in open(
        'eda/docking/splits/posebusters_pdb_ccd_ids.txt').readlines()])
    pose_df = df[df[target_key].isin(posebuster_v2_pdbs)]
    if len(pose_df):
        main(pose_df, name="Posebuster v2")
