import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime
from data.data_utils import pmap
from data.data_stucture.lmdb_dataset import LmdbPPI
from data.data_stucture.lmdb_dataset import Subset


class PPIData:
    def split(self, data, df, train_ratio=.7, valid_ratio=.1):
        ###
        # Drop specific cluster first
        # df = df[df['isbound'] == True]
        new_df = df[df['cluster'] != '20']
        print("[PPIData] drop cluster 20 from the dataset.")
        # new_df = df[(df['dockq'] < 0.25) | (df['dockq'] > 0.8)]  # performance not good, worse than training on >0.23
        print(f"[PPIData] after filer rules-> {len(new_df)}/{len(df)}")
        df = new_df
        ###
        # Split by time, indices of df can't be changed!!
        n = len(df)
        release_time = [[datetime.strptime(row['released_date'], "%Y/%m/%d"), i] for i, row in df.iterrows()]
        release_time = sorted(release_time, key=lambda x: x[0])
        indices = [x[1] for x in release_time]
        train_indices = indices[:int(train_ratio * n)]
        valid_indices = indices[int(train_ratio * n): int((train_ratio + valid_ratio) * n)]
        test_indices = indices[int((train_ratio + valid_ratio) * n):]
        # delete repeat uniprot from valid/test
        train_uniprot = df.loc[train_indices][self.target_key].unique().tolist()
        valid_df = df.loc[valid_indices]
        valid_indices = valid_df[~valid_df[self.target_key].isin(train_uniprot)].index.tolist()
        test_df = df.loc[test_indices]
        test_indices = test_df[~test_df[self.target_key].isin(train_uniprot)].index.tolist()
        # added the dropped samples back to training set
        train_indices.extend(valid_df[valid_df[self.target_key].isin(train_uniprot)].index.tolist())
        train_indices.extend(test_df[test_df[self.target_key].isin(train_uniprot)].index.tolist())
        # assign
        train, valid, test = Subset(data, train_indices), Subset(data, valid_indices), Subset(data, test_indices)
        ###
        # log
        # Valid Date:2021-01-27 00:00:00, Test Date:2022-02-16 00:00:00
        print(
            f"Valid Date:{release_time[int(train_ratio * n)][0]}, "
            f"Test Date:{release_time[int((train_ratio + valid_ratio) * n)][0]}")
        print(f"train:{len(train)}/valid:{len(valid)}/test:{len(test)}")
        # Debug
        df.loc[train_indices, 'Group'] = 'Training'
        df.loc[valid_indices, 'Group'] = 'Validation'
        df.loc[test_indices, 'Group'] = 'Testing'
        df.to_csv('/mnt/superCeph2/private/user/revoli/pp_docking/ppi-data/tmp_ppi/debug.csv', index=False)
        return train, valid, test

    def __init__(self, args, istrain=True, n_jobs=1):
        print(f"# [PPIData] Dataloader: lmdb, n_jobs={n_jobs}")
        self.args = args
        self.n_jobs = n_jobs
        # Cached Folder
        self.cached_folder = f"{args['work_path']}/tmp_ppi"
        base_name = os.path.basename(args['data_path'])
        os.makedirs(self.cached_folder, exist_ok=True)
        cached_path = f"{self.cached_folder}/{base_name}_{args['method']}"
        # csv
        self.label_key = 'isnative'
        self.target_key = 'cluster'  # Uniprot <-Not correct
        df = pd.read_csv(args['data_path'])
        if self.target_key in df:
            df = df.astype({self.target_key: str})
        # Debug
        ##
        data = self._pre_complex(df, cached_path)
        if istrain:
            self.datasets = self.split(data, self.df)
        else:
            self.datasets = data

    def _split_query(self, query):
        queries = []
        step = 100000
        if len(query) > step:
            for i in range(0, len(query), step):
                queries.append(query[i:i + step])
        else:
            queries.append(query)
        return queries

    def _read_pdb_files(self, df):
        pdb_folder = f"{self.args['work_path']}/{self.args['pocket_path']}"
        query = []
        for i, row in df.iterrows():
            ###
            HL = row['Hchain'] + row['Lchain']
            ab_chains = ''.join(str(row['antigen_chain']).replace(" ", "").split('|'))
            pdb_prefix = f"{row['pdb']}_{HL}_{ab_chains}"  # pdb_HL_A_0
            ##
            # Hard Code for now
            if 'isbound' in row:
                bound_prefix = 'bound' if row['isbound'] else 'unbound'
                pdb_f = f"{pdb_folder}/{bound_prefix}/pred1/convert/{pdb_prefix}/{pdb_prefix}_{row['pose']}.pdb"
            else:
                # task2_1_abs_127_scho/task2_1_abs_127_scho_0.pdb
                pdb_prefix = f"{row['pdb']}"
                pdb_f = f"{pdb_folder}/{pdb_prefix}/{pdb_prefix}_{row['pose']}.pdb"
                # for top100 file
                # pdb_f = [pdb_f, f"{pdb_folder}/{pdb_prefix}/{pdb_prefix}_antibody.pdb"]
            ###
            row = pdb_f, row['Hchain'] + ' | ' + row['Lchain'], row['antigen_chain']
            query.append(row)
        # make feature by using featurizer
        queries = self._split_query(query)
        complex_data_all = []
        for query in queries:
            complex_data = pmap(self.args['complex_to_data'],
                                query,
                                node_featurizer=self.args['node_featurizer'],
                                edge_featurizer=self.args['edge_featurizer'],
                                debug=0,
                                n_jobs=self.n_jobs)
            complex_data_all.extend(complex_data)
        return complex_data_all

    def _write_dataset(self, data, df, cached_path):
        LmdbPPI.make_shared(data, cached_path)
        # Save filtered df
        csv_path = cached_path + '.csv'
        df.to_csv(csv_path, index=False)

    def _get_dataset(self, cache_path):
        dataset = LmdbPPI(cache_path, label_key=self.label_key, target_key=self.target_key)
        # refresh df
        csv_path = cache_path + '.csv'
        self.df = pd.read_csv(csv_path)
        return dataset

    def _check_exists(self, cache_path):
        f = glob.glob(cache_path)
        ###
        if len(f):
            print(f'Loading previously saved cached file. <-- {cache_path}')
            return True
        print(f"# [PPIData] Cached file not exists:{cache_path}")
        return False

    def _assign_label2x(self, x, df):
        for i, row in enumerate(x):
            # assign label
            if self.label_key in df:
                row[self.label_key] = df.iloc[i][self.label_key]
            # assign target-pdb for now
            if self.target_key in df:
                row[self.target_key] = df.iloc[i][self.target_key]
        return x

    def _pre_complex(self, df, cached_path):
        # 0. Reload
        if self.args['reload'] and self._check_exists(cached_path):
            wdata = self._get_dataset(cached_path)
            return wdata
        #####
        print(f"# [PPIData] Making complex data {len(df)}, Method:{self.args['method']}")
        complex_data = self._read_pdb_files(df)
        # Check validation on data
        valid_ids, data = [], []
        for i, g in enumerate(complex_data):
            if g is not None:
                valid_ids.append(i)
                data.append(g)
        print(f"@ [PPIData] Valid Samples:{len(valid_ids)} / {len(complex_data)}")
        df = df.iloc[valid_ids].reset_index(drop=True)
        # write the label
        data = self._assign_label2x(data, df)
        #
        self._write_dataset(data, df, cached_path)
        wdata = self._get_dataset(cached_path)
        print('\n' + '+' * 100)
        return wdata
