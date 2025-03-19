import argparse
import os
import warnings

import torch

from utils.configure import model_list, get_exp_configure
from utils.train_utils import get_model_arg

warnings.filterwarnings("ignore", category=FutureWarning)


def set_random_seed(seed=0):
    import random
    import numpy as np
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description='SBDD Prediction')
    parser.add_argument('-model', type=str,
                        choices=model_list,
                        default='Interformer', help='Model to use')
    parser.add_argument('-debug', default=False, action='store_true')
    # Common Data
    parser.add_argument('-data_path', type=str, required=False, default="", help='train csv file.')
    parser.add_argument('-work_path', default='/opt/home/revoli/data_worker', type=str, required=False)
    parser.add_argument('-split_folder', default='diffdock_splits', type=str, required=False)
    parser.add_argument('-ligand_folder', default='ligand', type=str, required=False)
    parser.add_argument('-pocket_path', type=str, required=False, default="pocket", help='pocket and ligand folder.')
    parser.add_argument('-use_mid', default=False, action='store_true')
    parser.add_argument('-affinity_pre', default=False, action='store_true',
                        help='Whether to use affinity_max_amino parameters '
                             'for collecting amino atoms, it is only be used in affinity mode.')
    #
    parser.add_argument('-batch_size', type=int, default=6, required=False)
    parser.add_argument('-worker', type=int, default=8, required=False, help="Number of worker for data loader.")
    parser.add_argument('-n_jobs', type=int, default=70, required=False, help="Number of worker for data loader.")
    parser.add_argument('-method', type=str, default='Gnina2', required=False, help='Choose Method')
    parser.add_argument('-Code', type=str, default='Flower', required=False, help='Training Code')
    parser.add_argument('-dataset', type=str, default='sbdd', required=False, help='Dataset Type.')
    #####################
    # Training
    parser.add_argument('-seed', type=int, default=8888, required=False)
    parser.add_argument('-clip', type=float, default=5., required=False)
    parser.add_argument('-checkpoint', default='./', type=str, help="Location of model to save.")
    parser.add_argument('-gpus', type=int, default=1, required=False, help="the num of gpus use in one machine")
    parser.add_argument('-patience', type=int, default=10, required=False)
    parser.add_argument('-early_stop_metric', type=str, default='val_loss', required=False)
    parser.add_argument('-early_stop_mode', type=str, default='min', required=False)
    parser.add_argument('-num_epochs', type=int, default=2000, required=False)
    parser.add_argument('-main_loop', type=int, default=1, required=False)
    parser.add_argument('-precision', type=int, default=16, required=False)
    # sampler
    parser.add_argument('-per_target_sampler', type=int, default=0, required=False)
    parser.add_argument('-target_wise_sampler', type=int, default=0, required=False)
    parser.add_argument('-native_sampler', type=int, default=0, required=False)
    # filtering
    parser.add_argument('-filter_type', default='normal', required=False, type=str)
    # NN-docking
    parser.add_argument('-use_ff_ligands', type=str, default='uff', required=False)
    parser.add_argument('-num_ff_mols', type=int, default=1, required=False)
    ####################
    # Loss
    parser.add_argument('-neg_weight', type=float, default=1., required=False)
    # DDP
    parser.add_argument('-num_nodes', type=int, default=1, required=False)
    parser.add_argument('-num_gpus', default=8, type=int)
    # Metric
    parser.add_argument('-per_target', default=False, action='store_true', required=False, help='Per-target Pearson')
    parser.add_argument('-metric_name', default=['r'], required=False, help='metric method')
    #################
    # Inference
    parser.add_argument('-test_csv', default=['./test.csv'], nargs='+', help="All test.csv")
    parser.add_argument('-reload', default=True, action='store_false', required=False, help='Turn off reload')
    parser.add_argument('-ensemble', default=['checkpoint'], nargs='+', help="Ensemble List")
    parser.add_argument('-inference', default=False, action='store_true', required=False)  # just for record
    parser.add_argument('-posfix', default="*val_loss*", type=str, required=False)  # just for record
    parser.add_argument('-energy_output_folder', default="", type=str,
                        required=False)  # just for record
    parser.add_argument('-uff_as_ligand', default=False, action='store_true', required=False,
                        help='Switch to uff as the reference ligand to do predictions,'
                             ' using it while uff ligand are not the same with reference ligands.')
    ####
    # JIZHI
    import json
    WORKSPACE_PATH = os.environ.get('JIZHI_WORKSPACE_PATH', '')
    if WORKSPACE_PATH:
        text = open(WORKSPACE_PATH + '/job_param.json', 'r').read()
        jizhi_json = json.loads(text)
        # grep model_name first
        model_name = jizhi_json.get('model', model_list[0])
        parser = get_model_arg(parser, model_name)
        args = parser.parse_args().__dict__
        # refresh argument by jizhi config
        for key in jizhi_json:
            args[key] = jizhi_json[key]
        # Open Jizhi Reporter
        args['jizhi'] = True
    else:
        # Get Model argument
        parser = get_model_arg(parser)
        args = parser.parse_args().__dict__
        args['jizhi'] = False
    # Configure
    args['exp'] = '_'.join([args['model'], args['method']])
    args.update(get_exp_configure(args['exp']))
    #####################
    # Hard Code
    # device, default is using all available gpus
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUS:{n_gpus}")
    if n_gpus == 0:
        print("using CPU now")
        args['gpus'] = 'cpu'
    set_random_seed(args['seed'])
    print(f"# Using GPUS:{args['gpus']}")
    #####
    print(args)
    print('*' * 100)
    return args
