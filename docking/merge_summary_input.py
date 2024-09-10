# This script for mering the docking results and original input csv together
import sys
import pandas as pd

summary_df = pd.read_csv(sys.argv[1])
ori_df = pd.read_csv(sys.argv[2])

summary_df = summary_df.rename(columns={"pdb_id": "Target"})
# only taking the required columns
summary_df = summary_df[['Target', 'pose_rank', 'num_torsions', 'energy', 'rmsd']]
df = ori_df.merge(summary_df, on="Target", how='outer')

# failed to dock
print("Failed to docking, excluding...")
failed = df[df['pose_rank'].isna()]
print(failed)
print(f'n->{len(failed)}')
df = df[~df['pose_rank'].isna()]
print("=" * 100)
# Output
output_f = sys.argv[2][:-4] + '.round0.csv'
df.to_csv(output_f, index=False)
print(df)
print(f"Merged summary and original csv -> {output_f}")
