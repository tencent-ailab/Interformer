import os

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

from interformer.data.data_process import GraphDataModule
from interformer.utils.cluster import auto_configure_nccl
from interformer.utils.parser import get_args
from interformer.utils.train_utils import load_model, param_count, get_callbacks

print(f"# Torch Version:{torch.__version__}")


def main(args):
    # NCCL
    auto_configure_nccl()
    # Data Processing
    dm = GraphDataModule(args)
    # Model Setup
    model = load_model(args)
    param_count(model)
    # Start Training
    print('+' * 100)
    #
    print(f"# Precision:{args['precision']}f")
    # dataset_model
    folder_name = f"{os.path.basename(args['data_path'])[:-4]}_{args['model']}_{args['Code']}"
    print(f"#Folder_Name:{folder_name}")
    tb_logger = pl_loggers.TensorBoardLogger(f"{args['checkpoint']}/lightning_logs/", folder_name)
    trainer = pl.Trainer(
        devices='auto',
        precision=args['precision'],
        max_epochs=args['num_epochs'],
        log_every_n_steps=20,
        fast_dev_run=False,
        callbacks=get_callbacks(args),
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        num_sanity_val_steps=0,  # num of batches in val, to check, -1 means the whole val
        accelerator='cuda',
        default_root_dir=args['checkpoint'],
        logger=tb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler=False,  # it is important, make sure trainner not using their own sampler
        # reload_dataloaders_every_n_epochs=1,
        num_nodes=args['num_nodes'],
    )
    trainer.fit(model, datamodule=dm)
    # Final Test
    print(f"# Testing by:{trainer.checkpoint_callback.best_model_path}")
    test_result = trainer.test(model, ckpt_path='best', datamodule=dm)
    print(test_result)
    print("+" * 100)
    print("*********END of One Model*******")
    return test_result


if __name__ == "__main__":
    args = get_args()
    # Main loop
    total = []
    for i in range(args['main_loop']):
        print(f"# Main Loop->{i}")
        total.append(main(args))
    print("DONE")
    dist.destroy_process_group()
