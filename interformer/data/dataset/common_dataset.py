import glob
import pandas as pd
from data.data_stucture.lmdb_dataset import LmdbPPI


class Dataset:
    def __len__(self):
        return len(self.df)

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
            print(f'[Dataset] Loading previously saved cached file. <-- {cache_path}')
            return True
        print(f"# [Dataset] Cached file not exists:{cache_path}")
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
