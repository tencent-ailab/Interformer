import pandas as pd
import matplotlib.pyplot as plt


def get_homology_data(name):
    data = [x.split('\t') for x in open(f'{a8_folder}/{name}.m8').readlines()]

    data_map = {}
    for item in data:
        # grep maximum seq identity
        query_p, score = item[0], float(item[2])
        if query_p not in data_map:
            data_map[query_p] = score  # retrieve the top row
        # else:
        #   data_map[query_p] = max(score, data_map[query_p])  # find the maximum seq_identity
    print(f"Searched num_target:{len(data_map)}")
    return data_map


def eval(df, name, top=1):
    name = name + f'-{top}'
    top1 = df.groupby('Target').head(top)
    top1 = top1.groupby('Target').min()
    print(f"N:{len(top1)}")
    ac = (top1['rmsd'] < 2.).sum() / len(top1)
    print(name, ac)
    print(name + '_median', top1['rmsd'].median())
    return ac


def load_docking_df(dock_df, key):
    # Sorting
    if key == 'pred_pose':
        dock_df = dock_df.sort_values('pred_pose', ascending=0)
    elif key == 'pose_rank':
        dock_df = dock_df.sort_values('pose_rank', ascending=1)
    #
    print(f"Dock num_target:{len(dock_df['Target'].unique())}")
    return dock_df


def get_bar_data(data_map, name):
    print(f"name:{name}")
    if name == 'Interformer-PoseScore':
        dock_df = pd.read_csv(inter_f)
        dock_df = dock_df[dock_df['pose_rank'] != 20]  # exclude crystal
        key = 'pred_pose'
    elif name == 'Interformer-Energy':
        dock_df = pd.read_csv(inter_f)
        dock_df = dock_df[dock_df['pose_rank'] != 20]  # exclude crystal
        key = 'pose_rank'
    elif name == 'DiffDock':
        dock_df = pd.read_csv(diff_f)
        key = 'pose_rank'
    elif name == 'DeepDock':
        dock_df = pd.read_csv(deep_f)
        missing = {'Target': '6n4b', 'pose_rank': 0, 'rmsd': 5.}  # can not be run by deepdock
        dock_df = dock_df._append(missing, ignore_index=True)
        key = 'pose_rank'
    #
    dock_df = dock_df[dock_df['Target'].isin(pdb_list)]
    ######
    dock_df = load_docking_df(dock_df, key)
    df = pd.DataFrame(list(data_map.items()), columns=['pdb', 'max_seq_identity'])
    # merge two pdb together
    df = df[df['pdb'].isin(dock_df['Target'].unique().tolist())]
    diff = list(set(dock_df['Target'].unique().tolist()) - set(df['pdb'].unique().tolist()))
    diff = [[x, 0.] for x in diff]
    diff = pd.DataFrame(diff, columns=['pdb', 'max_seq_identity'])
    df = pd.concat([df, diff])
    #
    ranges = [[-1., 0.3], [0.3, 0.95], [0.95, 1.0]]
    # ranges = [[-1, 1.0]]
    bar_data = []
    for start, end in ranges:
        tmp = df[(df['max_seq_identity'] > start) & (df['max_seq_identity'] <= end)]
        # for logging
        if start == -1.:
            start = 0.
        pdbs_list = tmp['pdb'].to_list()
        tmp = dock_df[dock_df['Target'].isin(pdbs_list)]
        ac = eval(tmp, f'({start}-{end}]')
        bar_data.append([name, f'({start}, {end}],\nN={len(tmp['Target'].unique())}', ac * 100.])
        print('=' * 100)
    # Debug
    # df.hist('max_seq_identity')
    # plt.show()
    bar_data = pd.DataFrame(bar_data, columns=['Method', x_axis_key, y_axis_key])
    return bar_data


if __name__ == '__main__':
    import seaborn as sns

    x_axis_key, y_axis_key = 'Homology cutoffs', '% RMSD < 2Å'
    root = '/opt/home/revoli/data_worker/paper/benchmark/docking/similarity/sequence'
    inter_f = f'{root}/core_timetest.round0_ensemble.csv'
    diff_f = f'{root}/diffdock_obrms_rmsd.csv'
    deep_f = f'{root}/deepdock_obrms_rmsd.csv'
    a8_folder = f'{root}/a8_files'
    pdb_list = [x.strip() for x in open(f'{root}/timesplit_test_sanitizable').readlines()]
    # Drawing
    data_map = get_homology_data('test')
    inter_bar_data = get_bar_data(data_map, 'Interformer-PoseScore')
    energy_bar_data = get_bar_data(data_map, 'Interformer-Energy')
    diff_bar_data = get_bar_data(data_map, 'DiffDock')
    deep_bar_data = get_bar_data(data_map, 'DeepDock')
    bar_data = pd.concat([inter_bar_data, energy_bar_data, diff_bar_data, deep_bar_data])
    sns.set(font_scale=1.5)
    sns.set_style("white")
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)
    colors = ['#36802d', '#c9df8a', '#f9d62e', '#fc913a']
    ax = sns.barplot(x=x_axis_key, y=y_axis_key, data=bar_data, hue='Method', edgecolor='black', dodge=True, width=.4,
                     palette=colors, saturation=1., legend=True,
                     alpha=.8)
    plt.legend(title='Method', loc='upper left')
    ax.set_ylim(0, 100)
    fig.savefig('/opt/home/revoli/data_worker/figures/3b.svg')
    plt.show()
