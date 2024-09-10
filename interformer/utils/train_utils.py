import glob
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging

from model.ppi_model.ppi_rescorer import PPIScorer
from model.transformer.graphormer.interformer import Interformer
from utils.configure import get_exp_configure


def get_model_arg(parser, model=''):
    model_name = model if model else parser.parse_known_args()[0].model
    if model_name == 'Interformer':
        parser = Interformer.add_model_specific_args(parser)
    elif model_name == 'PPIScorer':
        parser = PPIScorer.add_model_specific_args(parser)

    return parser


def load_model(args):
    model_name = args['model']
    if model_name == 'Graphformer':
        model = Graphormer(args)
    elif model_name == 'Beta':
        model = Beta(args)
    elif model_name == 'Interformer':
        model = Interformer(args)
    elif model_name == 'PPIScorer':
        model = PPIScorer(args)
    return model


def load_from_checkpoint(args, checkpoint):
    print("# Using Model <-", checkpoint)
    # get model's name from hparam first
    hparam = yaml.load(open('/'.join(checkpoint.split('/')[:-2]) + '/hparams.yaml'), Loader=yaml.Loader)
    model_name = hparam['args']['model']
    #
    if model_name == 'Interformer':
        model = Interformer.load_from_checkpoint(checkpoint)
    elif model_name == 'PPIScorer':
        model = PPIScorer.load_from_checkpoint(checkpoint)
    #
    print(model.hparams)
    return model


def param_count(model, print_model=True):
    if print_model:
        print(model)
    # for name, param in model.named_parameters():
    #   print(name)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Number of parameters = {param_count:,}')


def get_modl_checkpoint(metric, dataset_name, mode='max'):
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        filename=dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=1,
        mode=mode,
        save_last=True,
    )
    return checkpoint_callback


def get_callbacks(args):
    # Checkpoint
    checkpoints = []
    dataset_name = os.path.basename(args['data_path'])[:-4]
    # common log
    checkpoints.append(get_modl_checkpoint('val_loss', dataset_name, mode='min'))
    checkpoints.append(get_modl_checkpoint('val_affinity', dataset_name, mode='max'))
    if args['pose_sel_mode']:
        checkpoints.append(get_modl_checkpoint('val_pose_selection', dataset_name, mode='max'))
    # model specific
    if args['model'] == 'PPIScorer':
        checkpoints.append(get_modl_checkpoint('valid_auroc', dataset_name, mode='max'))

    ###
    # Early Stop
    early_stop_callback = EarlyStopping(
        monitor=args['early_stop_metric'],
        min_delta=0.0001,
        patience=args['patience'],
        verbose=False,
        mode=args['early_stop_mode'],
        check_finite=True  # if 0=stop training
    )
    lrm_callback = LearningRateMonitor(logging_interval='step')

    swa_callback = StochasticWeightAveraging(swa_lrs=0.01)

    return checkpoints + [early_stop_callback, lrm_callback, swa_callback]


def get_checkpoint_realpath(checkpoint_folder, posfix='*epoch*val_r*'):
    f = glob.glob(f"{checkpoint_folder}/checkpoints/{posfix}")
    if len(f) == 0:
        return ''
    return f[0]


def refresh_args(args, model, finetune=False):
    hparams = model.hparams
    args['method'] = hparams.args['method']  # refresh args
    args['model'] = hparams.args['model']
    # get featurizer once again
    exp_dict = get_exp_configure(args['model'] + '_' + args['method'])
    args['node_featurizer'] = exp_dict['node_featurizer']  # refresh args
    args['edge_featurizer'] = exp_dict['edge_featurizer']  # refresh args
    args['complex_to_data'] = exp_dict['complex_to_data']
    # override model parameters
    args['pose_sel_mode'] = hparams.args['pose_sel_mode'] if 'pose_sel_mode' in hparams.args else False
    args['energy_mode'] = hparams.args['energy_mode'] if 'energy_mode' in hparams.args else False
    if finetune:
        model.warmup_updates = args['warmup_updates']
        model.peak_lr = args['peak_lr']
        model.tot_updates = args['warmup_updates'] * 10
    if hasattr(model, 'VinaScoreHead'):
        if args['energy_output_folder']:
            model.VinaScoreHead.energy_output_folder = args['energy_output_folder']
    if 'affinity_pre' in hparams.args:
        args['affinity_pre'] = hparams.args['affinity_pre']

    return args


def auto_scale_batch(args, model, dm):
    trainer = pl.Trainer(
        gpus=[0],
        precision=32,
        max_epochs=args['num_epochs'],
        log_every_n_steps=10,
        fast_dev_run=True if args['debug'] else False,
        callbacks=get_callbacks(args),
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        num_sanity_val_steps=1,
        accelerator='ddp',
        default_root_dir=args['checkpoint'],
        replace_sampler_ddp=not args['per_target_sampler'],
        auto_scale_batch_size='power'  # it will be bug in multi-gpus
    )
    model.hparams.batch_size = 1
    reduce_value = 6
    result = trainer.tune(model, datamodule=dm)
    scale = result['scale_batch_size']
    print(f"Found Auto Batch RESULT:{scale}->{scale - reduce_value}")
    scale -= reduce_value
    # set it back
    # model.batch_size = scale
    # dm.batch_size = scale
    # trainer.gpus = model.hparams.args['gpus']
    # trainer.reset_train_dataloader(model)
    print("#" * 100)
    return scale
