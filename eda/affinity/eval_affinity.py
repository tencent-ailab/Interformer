import pandas as pd
import sys

if __name__ == '__main__':
    # this is a small evaluation on high confident poses
    df = pd.read_csv(sys.argv[1])
    print(f"n:{len(df)}")
    print(df.corr(numeric_only=True))
    if 'pred_pose' in df:
        print('=' * 100)
        # selecting top confident poses
        print("Only PoseScore > 0.8")
        df = df[df['pred_pose'] > 0.8]
        # total samples
        print(f"n:{len(df)}")
        print(df.corr(numeric_only=True))
