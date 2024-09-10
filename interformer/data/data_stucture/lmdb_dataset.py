import os.path
import pickle
import pandas as pd
import shutil
import lmdb
import numpy as np

from data.data_stucture.shard_dataset import ShardBinding


class Subset(object):
    """Subset of a dataset at specified indices

    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)


class LmdbPPI:
    def __init__(self, cache_file, label_key='isnative', target_key='UniProt', adj_label=False):
        print(f"@LMDB Loading <-{cache_file}")
        env = lmdb.open(
            cache_file + '/data.mdb',
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin(write=False)
        self.env = env
        self.txn = txn
        self.cache_file = cache_file
        self.keys = list(txn.cursor().iternext(values=False))
        self.len = 0
        self.label_key = label_key
        self.target_key = target_key
        self.adj_label = adj_label  # if the label is 1 or 0, adj them
        self.get_sampler_info()

    def get_sampler_info(self):
        info_csv = self.cache_file + '.csv'
        print(f"# [Lmdb] Geting Sampler Info <- {info_csv}")
        df = pd.read_csv(info_csv)
        if self.label_key in df:
            y = df[self.label_key].to_numpy()
            if self.adj_label:
                y = y * 7.5  # positive > 6. and negative should be negatives
                y[y == 0] = -2.
            self.labels = y
            self.targets = df[self.target_key].tolist() if self.target_key in df else df[
                'Target'].tolist()  # for pdbs temporally
        print("# [Lmdb] Finished Loading Sampler Info.")

    @staticmethod
    def stats_data(data):
        ligand_lens = []
        pocket_lens = []
        complex_lens = []
        for i, x in enumerate(data):
            ligand_len, pocket_len = x['lens']
            ligand_lens.append(ligand_len)
            pocket_lens.append(pocket_len)
            complex_lens.append(len(x['ndata']))

        def get_max_mean_min(x):
            return f":max={max(x)},mean={np.mean(x):4f},median={np.median(x):4f},std={np.std(x):4f},min={np.min(x)}"

        print(
            f"@ Data Size={len(data)}"
            f", ligand_len{get_max_mean_min(ligand_lens)}"
            f", pocket_len{get_max_mean_min(pocket_lens)}, "
            f"complex_len{get_max_mean_min(complex_lens)}")

    @staticmethod
    def make_shared(data, output_path):
        LmdbPPI.stats_data(data)
        if os.path.exists(output_path):
            print(f"# [Lmdb] Remove previous exists<-{output_path}")
            shutil.rmtree(output_path)
        # root
        root_path = os.path.dirname(output_path)
        if not os.path.exists(root_path):  # root folder not exists
            print(f"# creating root path<-{root_path}")
            os.makedirs(root_path, exist_ok=True)
        # write
        env = lmdb.open(output_path, map_size=1099511627776)
        txn = env.begin(write=True)
        for i, row in enumerate(data):
            row_pickle = pickle.dumps(row)
            byte_i = str(i).encode()
            txn.put(key=byte_i, value=row_pickle)
        txn.commit()
        env.close()

    def __getitem__(self, id):
        byte_idx = str(id).encode()
        X = pickle.loads(self.txn.get(byte_idx))
        return f"Complex:{id}", X, id

    def __len__(self):
        return len(self.keys)
