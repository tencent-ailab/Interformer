import glob
import os
import subprocess
import pandas as pd

root = '/opt/home/revoli/data_worker/interformer/train'
input_df = 'general_PL_2020_round0_full.csv'
df = pd.read_csv(f'{root}/{input_df}')
test_pdbs = [x.strip() for x in open(root + '/diffdock_splits/timesplit_test').readlines()]
core = [x.strip() for x in open(root + '/diffdock_splits/coresetlist').readlines()]
test_df = df[df['Target'].isin(test_pdbs)].copy()
core_df = df[df['Target'].isin(core)].copy()
test_df['task'] = 'test'
core_df['task'] = 'core'
split_df = pd.concat([test_df, core_df])
##########
print(f'Time n:{len(test_df[test_df['pose_rank'] == 0])}')
print(f'Core n:{len(core_df[core_df['pose_rank'] == 0])}')
# find out those extra target
full_time = set(test_df['Target'].unique().tolist())
core_set = set(core_df['Target'].unique().tolist())
test_pdbs = set(
    [os.path.basename(x)[:4] for x in glob.glob('/opt/home/revoli/eva/Interformer/energy_timetest/ligand/*')])
test_pdbs = test_pdbs - core_set
print(f'exceeded:{full_time - test_pdbs}, {test_pdbs - full_time}')
# save csv
split_df.to_csv(root + '/../test/core_timetest_round0.csv', index=False)

print("DONE")
