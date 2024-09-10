# It is a demo script for predicting single sample, but it is not TODO:
# It is not completed yet, we will do it later
import torch
from utils.parser import get_args
import os

from utils.train_utils import get_checkpoint_realpath


def setup():
    args = get_args()
    args['inference'] = True
    args['device'] = torch.device(args['gpus'])
    args['num_ff_mols'] = 1
    os.makedirs('result', exist_ok=True)
    return args


def infer(args):
    for checkpoint_folder in args['ensemble']:
        checkpoint = get_checkpoint_realpath(checkpoint_folder, posfix=args['posfix'])
        if checkpoint == '':
            print(f"# Model doesn't Exists:{checkpoint_folder}")
            continue


if __name__ == '__main__':
    # TODO: this script is for single complex prediction
    args = setup()
    infer(args)
