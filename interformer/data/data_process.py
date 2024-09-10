# it is the main data_process script, we have the data module for pytorch lightning..........
# this for DataModule script
from data.sampler import per_target_balance_sampler, LISA_sampler, Fullsampler
from data.dataset.bindingdata import BindingData
from data.dataset.ppi_dataset import PPIData
import torch
from functools import partial
from torch.utils.data import distributed
from pytorch_lightning import LightningDataModule
from data.collator.inter_collate_fn import interformer_collate_fn
from data.collator.ppi_collate_fn import ppi_collate_fn, ppi_residue_collate_fn


def sel_collate_fn(model):
    if model == 'Interformer':
        return interformer_collate_fn
    elif model == 'PPIScorer':
        return ppi_residue_collate_fn
    return interformer_collate_fn


def data_loader_warper(dataset, bs, model, energy_mode=False, num_workers=5, shuffle=False):
    collate_fn = partial(sel_collate_fn(model), energy_mode=energy_mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate_fn, num_workers=num_workers,
                                         shuffle=shuffle)
    return loader


class GraphDataModule(LightningDataModule):

    def __init__(self, args, istrain=True):
        super().__init__()
        print(f"# Dataset: {args['data_path']}")
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = 0 if args['debug'] else 66
        self.args = args
        mode = args['energy_mode'] if 'energy_mode' in args else False
        self.collate_fn = partial(sel_collate_fn(args['model']), energy_mode=mode)
        self.istrain = istrain

    def setup(self, stage=None):
        if self.args['dataset'] == 'PPI':
            bind_dataset = PPIData(self.args, n_jobs=1 if self.args['debug'] else self.args['n_jobs'],
                                   istrain=self.istrain)
        else:
            bind_dataset = BindingData(self.args, n_jobs=1 if self.args['debug'] else self.args['n_jobs'],
                                       istrain=self.istrain)
        if self.istrain:
            self.bind_train, self.bind_valid, self.bind_test = bind_dataset.datasets
        else:
            self.bind_test = bind_dataset.datasets
        # Unit-Test
        # for j in self.bind_train:
        #   print(',', end='')
        # for j in self.bind_train:
        #   print(',', end='')

    def train_dataloader(self):
        print("[GraphData] train loader")
        ######################
        batch_size = self.batch_size
        dataset = self.bind_train
        # This method may be abandoned, it may have bugs
        if self.args['per_target_sampler']:
            sampler = per_target_balance_sampler(dataset, batch_size=batch_size, seed=self.args['seed'],
                                                 num_samples=self.args['per_target_sampler'])
        elif self.args['native_sampler']:
            print('# We are using native ddp sampler now')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=self.args['seed'])
        else:
            # Batch-filling Target-wise & Neg Samples support
            sampler = Fullsampler(dataset, batch_size=batch_size, seed=self.args['seed'],
                                  num_samples=self.args['per_target_sampler'])

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False,
                                           collate_fn=self.collate_fn,
                                           sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        print("[GraphData] valid loader")
        dataset = self.bind_valid
        batch_size = self.batch_size
        if self.args['native_sampler']:
            print('# We are using native ddp sampler now')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = Fullsampler(dataset, batch_size=batch_size, seed=self.args['seed'],
                                  num_samples=self.args['per_target_sampler'])
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False,
                                           collate_fn=self.collate_fn, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        print("[GraphData] test loader")
        dataset = self.bind_test
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                                  num_workers=self.num_workers)
        return data_loader
