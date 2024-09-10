import numpy as np
import torch
import math
import os
import random
from itertools import cycle
from collections import defaultdict


class ShardBinding:

    def __init__(self, shared_list):
        self.pos_shared_list = sorted([x for x in shared_list if '.pos.' in x])
        self.neg_shared_list = sorted([x for x in shared_list if '.neg.' in x])
        self.shared_iter = cycle(self.pos_shared_list)
        self.neg_shared_iter = cycle(self.neg_shared_list)
        self.pos_X, self.pos_labels = [], []
        self.neg_X, self.neg_labels = [], []

    def load_one(self, IsPos=True):
        shared = next(self.shared_iter) if IsPos else next(self.neg_shared_iter)
        base = os.path.basename(shared)
        print(f"@Shared loading<-{base}")
        data = torch.load(shared)
        if IsPos:
            self.pos_X, self.pos_labels = data
        else:
            self.neg_X, self.neg_labels = data

    def loop_shared(self, balance=False):
        if len(self.pos_shared_list) and (not len(self.pos_X) or len(self.pos_shared_list) != 1):
            self.load_one()
        if len(self.neg_shared_list) and (not len(self.neg_X) or len(self.neg_shared_list) != 1):
            self.load_one(False)
        # cat pos and neg
        if balance and len(self.neg_X) > len(self.pos_X):
            print(f"# Balancing Neg:{len(self.neg_X)}->Pos:{len(self.pos_X)}")
            neg_indices = random.sample(list(range(len(self.neg_X))), len(self.pos_X))
            self.neg_X = [self.neg_X[i] for i in neg_indices]
            self.neg_labels = [self.neg_labels[i] for i in neg_indices]
        #
        self.X = self.pos_X + self.neg_X
        self.labels = self.pos_labels + self.neg_labels
        self.targets = [row[-1] for row in self.X]

    @staticmethod
    def stats_data(data):
        ligand_lens = []
        pocket_lens = []
        complex_lens = []
        for i, (x, y) in enumerate(data):
            ligand_len, pocket_len = x['lens']
            ligand_lens.append(ligand_len)
            pocket_lens.append(pocket_len)
            complex_lens.append(len(x['ndata']))

        def get_max_mean_min(x):
            return f":max={max(x)},mean={np.mean(x):4f},std={np.std(x):4f},min={np.min(x)}"

        print(
            f"@ Data Size={len(data)}, "
            f"ligand_len{get_max_mean_min(ligand_lens)}, pocket_len{get_max_mean_min(pocket_lens)}, "
            f"complex_len{get_max_mean_min(complex_lens)}")

    @staticmethod
    def write_shared(X, labels, output_file_path, shared_id):
        torch.save([X, labels], output_file_path)
        X, labels = [], []
        shared_id += 1
        return X, labels, shared_id

    @staticmethod
    def make_shared(data, output_path, max_count=40000, isInference=False, no_neg=False, balancePosNeg=False):
        # statistic data first
        ShardBinding.stats_data(data)
        # split them by target
        cluster_data = defaultdict(lambda: defaultdict(list))
        for i, (x, y) in enumerate(data):
            target = x[-1]
            if y > 0:
                cluster_data[target]['pos'].append((x, y))
            else:
                cluster_data[target]['neg'].append((x, y))
        #
        # gathering data
        X, labels, neg_X, neg_labels = [], [], [], []
        shared_id, neg_shared_id = 0, 0
        i = 0
        for t, data in cluster_data.items():
            # Pos
            pos_len = len(data['pos'])
            x = [m[0] for m in data['pos']]
            y = [m[1] for m in data['pos']]
            X.extend(x)
            labels.extend(y)
            # Negatives samples
            if not no_neg and data['neg']:
                x = [m[0] for m in data['neg']]
                y = [m[1] for m in data['neg']]
                # balance the same amount of pos&neg
                if balancePosNeg:
                    idx = np.random.choice(list(range(len(x))), pos_len)
                    x = [x[i] for i in idx]
                    y = [y[i] for i in idx]
                # If Inference put neg_samples to pos_X
                if isInference:
                    X.extend(x)
                    labels.extend(y)
                else:
                    neg_X.extend(x)
                    neg_labels.extend(y)
            # display
            if i % 50 == 0:
                print(f"{i}/{len(cluster_data)}", end=',')
            i += 1
            # writing
            if len(X) >= max_count:
                X, labels, shared_id = ShardBinding.write_shared(X, labels, output_path + f'-{shared_id:06d}.pos.bin',
                                                                 shared_id)
                neg_bin_f = output_path + f'-{neg_shared_id:06d}.neg.bin'
                neg_X, neg_labels, neg_shared_id = ShardBinding.write_shared(neg_X, neg_labels,
                                                                             neg_bin_f,
                                                                             neg_shared_id)
        # tailing
        if len(X):
            torch.save([X, labels], output_path + f'-{shared_id:06d}.pos.bin')
        if len(neg_X):
            torch.save([neg_X, neg_labels], output_path + f'-{neg_shared_id:06d}.neg.bin')

    def __getitem__(self, item):
        return f"Complex:{item}", self.X[item], self.labels[item].view(-1), item

    def __len__(self):
        return len(self.labels)
