import random

import torch.distributed as dist
import torch
import math
import numpy as np
from collections import defaultdict
import itertools


def odd2even(x):
    if x % 2 == 0:
        return x
    return x + 1


class LISA_sampler(torch.utils.data.Sampler):

    def __init__(self, data_source, seed=0, num_samples=20000):
        # dist
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.seed = seed
        self.epoch = 0
        # epoch num
        self.epoch_num = num_samples * 2
        self.num_samples = odd2even(math.ceil((self.epoch_num - self.num_replicas) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # works
        target = data_source.targets
        labels = data_source.labels
        self.data_source = data_source
        self.uni_targets = np.unique(target)
        self.labels = labels
        threshold = 0.
        # create target to index list
        self.t2idx = dict((t, [[], []]) for t in self.uni_targets)
        num_pos, num_neg = 0, 0
        for i, t in enumerate(target):
            if labels[i] < threshold:
                self.t2idx[t][0] += [i]
                num_neg += 1
            else:
                self.t2idx[t][1] += [i]
                num_pos += 1
        # create label to index list
        max_pic50 = 12
        self.y_range = np.linspace(-max_pic50, max_pic50, max_pic50 * 10)
        self.y2idx = dict((v, defaultdict(list)) for v in self.y_range)
        for i, (t, y) in enumerate(zip(target, labels)):
            y = self._y2onehot(y)
            self.y2idx[y][t].append(i)

        # summary
        print(
            f"# Using LISA, Rank:{self.rank}, seed:{seed}, "
            f"size:{num_samples}, uni-target:{len(self.uni_targets)}, "
            f"Pos:{num_pos}, Neg:{num_neg}, total_size:{self.total_size}, num_samples:{self.num_samples}")

    def _y2onehot(self, y):
        y_onehot = self.y_range[np.argmin(np.abs(y - self.y_range))]
        return y_onehot

    def _get_new_sample(self, sample, t):
        # strategy_p = random.random()
        strategy_p = 0.
        y = self._y2onehot(float(self.labels[sample]))
        # Strategy I
        if strategy_p <= 0.5:
            # select same y but different domain
            same_y_list = self.y2idx[y]
            keys = list(same_y_list.keys())
            keys.remove(t)
            if keys:
                new_sample = np.random.choice(same_y_list[np.random.choice(keys)])
            else:
                # no other sample, merge its self
                new_sample = sample
        # Strategy II
        else:
            # select same domain different y
            new_sample = sample
            b2_y_list = self.t2idx[t][y >= 0.][:]
            np.random.shuffle(b2_y_list)
            for b2 in b2_y_list:
                if self._y2onehot(float(self.labels[b2])) != y:
                    new_sample = b2
                    break

        return new_sample

    def __iter__(self):
        # generating indices
        all = []
        # random generator
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while len(all) < self.total_size:
            # Sampling
            sel_targets = [self.uni_targets[i] for i in torch.randperm(len(self.uni_targets), generator=g).tolist()]
            rand_classes = torch.randint(2, [len(sel_targets)]).tolist()
            for i, t in enumerate(sel_targets):
                sel_class_samples = self.t2idx[t][rand_classes[i]]
                if sel_class_samples:
                    sample = np.random.choice(sel_class_samples)
                    all.append(sample)  # select one of the sample, based on target and class
                    # get new sample
                    new_sample = self._get_new_sample(sample, t)
                    all.append(new_sample)
        # dist samples
        all = all[self.rank * self.num_samples: self.rank * self.num_samples + self.num_samples]
        assert len(all) == self.num_samples
        yield from all

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SamplerBase(torch.utils.data.Sampler):

    def __init__(self, data_source, batch_size, seed=0):
        target = [str(data_source.dataset.targets[data_source.indices[i]]) for i in range(len(data_source))]
        labels = [data_source.dataset.labels[data_source.indices[i]] for i in range(len(data_source))]
        # dist
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        #
        self.data_source = data_source
        self.uni_targets = np.unique(target)
        self.labels = labels
        pos_neg_threshold = 0.
        pos_threshold = 6.
        # create target to index list
        self.t2idx = dict((t, [defaultdict(list), {'active': [], 'inactive': [], 'all': []}]) for t in self.uni_targets)
        # create pos, neg pools
        num_pos, num_neg = 0, 0
        for i, t in enumerate(target):
            i = int(i)
            # Negatives
            if labels[i] < pos_neg_threshold:
                self.t2idx[t][0][abs(labels[i])].append(i)  # key as Label
                num_neg += 1
            # Positives
            else:
                if labels[i] < pos_threshold:
                    self.t2idx[t][1]['inactive'].append(i)
                else:
                    self.t2idx[t][1]['active'].append(i)
                self.t2idx[t][1]['all'].append(i)
                num_pos += 1
        self.num_pos = num_pos
        self.num_neg = num_neg
        # epoch num
        num_samples = num_pos + num_neg
        self.epoch_num = min(num_samples, self.num_pos + self.num_neg)  # num_samples can't exceeds num_pos
        self.num_samples = math.ceil((self.epoch_num - self.num_replicas) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def display_info(self, name):
        # Debug
        print(
            f"# [SamplerBase] Using {name}, Seed:{self.seed}, "
            f"Rank:{self.rank}, num_uni-target:{len(self.uni_targets)}, "
            f"Pos:{self.num_pos}, Neg:{self.num_neg}, "
            f"Total_size:{self.total_size}, one_gpu_num_samples:{self.num_samples}")

    def neg_preprocess(self):
        # remove those unusual targets first
        if self.num_neg:
            num_samples = 0
            final_uni_targets = []
            for t in self.uni_targets:
                pos_pool, neg_pool = self.t2idx[t][1]['all'], self.t2idx[t][0]
                if len(pos_pool) and len(neg_pool):  # Only Positive Target are selected
                    final_uni_targets.append(t)
                    num_samples += len(pos_pool)
            print(
                f"# [SamplerBase] Pruning Neg-Only or "
                f"Pos-Only Target Out-> {len(final_uni_targets)}/{len(self.uni_targets)}")
            self.uni_targets = final_uni_targets
            # calculate num_samples
            num_samples = math.ceil((num_samples - self.num_replicas) / self.num_replicas)
            self.num_samples = num_samples * 2  # neg + pos
            self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    @staticmethod
    def int2list(a):
        if isinstance(a, int):
            return [a]
        return a.tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def CrossJoin2Lists(self, a, b):
        assert (len(a) == len(b))
        ab = []
        for i in range(len(a)):
            ab.append(a[i])
            ab.append(b[i])
        return ab

    def get_align_neg_samples(self, neg_pool, pos_samples):
        # select neg by aligning pos
        neg_samples = []
        y = [self.labels[x] for x in pos_samples]
        for m in y:
            if m in neg_pool and len(neg_pool):
                lucky_neg = np.random.choice(neg_pool[m])
            else:
                # no align negs align random neg
                rand_t = random.choice(list(neg_pool.keys()))
                lucky_neg = np.random.choice(neg_pool[rand_t])
            lucky_neg = int(lucky_neg)
            neg_samples.append(lucky_neg)
        return neg_samples


class per_target_balance_sampler(SamplerBase):

    def __init__(self, data_source, batch_size, seed=0, num_samples=20000):
        super(per_target_balance_sampler, self).__init__(data_source, batch_size, seed)
        self.neg_preprocess()
        # reset
        if self.num_neg:
            num_samples = num_samples / 2
            num_samples = math.ceil((num_samples + self.num_replicas) / self.num_replicas)
            self.num_samples = num_samples * 2
        else:
            self.num_samples = math.ceil((num_samples + self.num_replicas) / self.num_replicas)
        #
        self.total_size = self.num_samples * self.num_replicas
        # display
        self.display_info('PerTarget')

    def __iter__(self):
        # generating indices
        all_pos, all_neg = [], []
        g = torch.Generator()
        local_seed = self.seed + self.epoch
        g.manual_seed(local_seed)
        np.random.seed(local_seed)
        sel_targets = self.uni_targets[:]
        np.random.shuffle(sel_targets)
        print(f"# [PerTarget] local_seed:{local_seed}")
        # Sampling
        while len(all_pos) + len(all_neg) < self.total_size:
            for t in sel_targets:
                if self.num_neg:
                    # select 1 pos, 1 neg
                    pos_pool, neg_pool = self.t2idx[t][1]['all'][:], self.t2idx[t][0]
                    pos_samples, neg_samples = [], []
                    if len(pos_pool):
                        pos_samples.append(int(np.random.choice(pos_pool)))
                    if len(pos_pool) and len(neg_pool):
                        neg_samples = self.get_align_neg_samples(neg_pool, pos_samples)
                    # skip unbalanced cases
                    if pos_samples and neg_samples:
                        all_pos.extend(pos_samples)
                        all_neg.extend(neg_samples)
                        # check num of the samples
                        if len(all_pos) + len(all_neg) == self.total_size:
                            break
                else:  # positive_only case
                    pos_pool = self.t2idx[t][1]['all'][:]
                    all_pos.append(int(np.random.choice(pos_pool)))
                    if len(all_pos) == self.total_size:
                        break
        ###
        if self.num_neg:
            all_pos = all_pos[self.rank:self.total_size // 2:self.num_replicas]
            all_neg = all_neg[self.rank:self.total_size // 2:self.num_replicas]
            all = self.CrossJoin2Lists(all_pos, all_neg)
        else:
            all = all_pos[self.rank:self.total_size:self.num_replicas]
        # dist samples
        assert len(all) == self.num_samples
        print(f"# [PerTarget] Done Creating Indices:rank={self.rank},len={len(all)}")
        self.epoch += 1  # next epoch, it has to be manuel
        yield from all

    def __len__(self) -> int:
        return self.num_samples


class Fullsampler(SamplerBase):

    def __init__(self, data_source, batch_size, seed=0, num_samples=20000):
        super(Fullsampler, self).__init__(data_source, batch_size, seed)
        self.neg_preprocess()
        self.display_info("FullSampler")

    def __iter__(self):
        # generating indices
        all_pos, all_neg = [], []
        g = torch.Generator()
        local_seed = self.seed + self.epoch
        g.manual_seed(local_seed)
        np.random.seed(local_seed)
        sel_targets = self.uni_targets[:]
        np.random.shuffle(sel_targets)
        print(f"# [FullSampler] local_seed:{local_seed}")
        # Sampling
        for i, t in enumerate(sel_targets):
            pos_pool, neg_pool = self.t2idx[t][1]['all'][:], self.t2idx[t][0]
            # Task3
            if self.num_neg:
                pos_samples, neg_samples = [], []
                if len(pos_pool):
                    # take all critically important.. do not take only one sample from target
                    np.random.shuffle(pos_pool)
                    pos_samples = pos_pool
                if len(pos_pool) and len(neg_pool):
                    neg_samples = self.get_align_neg_samples(neg_pool, pos_samples)
                # skip unbalanced cases
                if pos_samples and neg_samples:
                    all_pos.extend(pos_samples)
                    all_neg.extend(neg_samples)
            else:
                np.random.shuffle(pos_pool)
                pos_samples = pos_pool
                all_pos.extend(pos_samples)
        # Output
        # print('BeforeDist', self.rank, f"N:{len(all_pos)}", all_pos[:10], '\n', '=' * 100)
        if self.num_neg:
            all_pos = all_pos[self.rank:self.total_size // 2:self.num_replicas]
            all_neg = all_neg[self.rank:self.total_size // 2:self.num_replicas]
            all = self.CrossJoin2Lists(all_pos, all_neg)
        else:
            all = all_pos[self.rank:self.total_size:self.num_replicas]
        # Assert
        for i, item in enumerate(all):
            if item is None:
                print('[FullSampler] error', i)
        assert len(all) == self.num_samples
        print(f"# [FullSampler] Done Creating Indices:rank={self.rank},len={len(all)}")
        self.epoch += 1  # next epoch, it has to be manuel
        yield from all

    def __len__(self) -> int:
        return self.num_samples
