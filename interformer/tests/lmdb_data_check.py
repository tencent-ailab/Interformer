from data.data_stucture.lmdb_dataset import LmdbPPI

# Check whether the data is good
if __name__ == '__main__':
    # cache_path = ('/opt/home/revoli/data_worker/interformer/'
    #               'poses/tmp_beta/general_PL_2020_round0_full.csv_Gnina2_full-use_ff_ligands')
    cache_path = '/opt/home/revoli/data_worker/interformer/poses/tmp_beta/general_PL_2020.csv_Gnina2_normal-use_ff_ligands'
    dataset = LmdbPPI(cache_path, label_key='pIC50', target_key='Target')
    # check whether it have consistent label
    data = []
    for i in range(len(dataset)):
        item = dataset[i]
        row = item[1]
        data.append(row)
    #     if data['pdb'] != data['Target']:
    #         print(data)
    # statistic
    LmdbPPI.stats_data(data)
    print("Passed Tested.")
