import pytorch_lightning as pl
from model.model_utils import PolynomialDecayLR
import torch


class PPI(pl.LightningModule):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.save_hyperparameters()

        # learning scheduler
        self.clip = args['clip']
        self.warmup_updates = args['warmup_updates']
        self.tot_updates = args['warmup_updates'] * 15  # for simplicity
        self.peak_lr = args['peak_lr']
        self.end_lr = args['end_lr']
        self.weight_decay = args['weight_decay']

    def training_step(self, batched_data, batch_idx):
        x, _ = batched_data
        y_hat, loss = self(x)
        # Clip
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        # Log
        self.log('train_clip_norm', norm, sync_dist=False)
        self.log('train_loss', loss, sync_dist=True)
        return {"loss": loss, "x": x, "y_hat": y_hat}

    def validation_step(self, batched_data, batch_idx):
        x, _ = batched_data
        y_hat, loss = self(x)  # forward
        ###
        # Metrics
        self.log('val_loss', loss, sync_dist=True)
        ###
        return {"loss": loss, "x": x, "y_hat": y_hat}

    def test_step(self, batched_data, batch_idx):
        x, _ = batched_data
        y_hat, loss = self(x)  # forward
        ###
        # Metrics
        self.log('test_loss', loss, sync_dist=True)
        return {"loss": loss, "x": x, "y_hat": y_hat}

    def training_epoch_end(self, outputs):
        # outputs = each device's all step gathering
        print(f"[train] At train epoch, global_step={self.global_step}")
        self.metric_fn(outputs, prefix='train')
        pass

    def validation_epoch_end(self, outputs):
        print(f"[train] At valid epoch, global_step={self.global_step}")
        self.metric_fn(outputs, prefix='valid')
        pass

    def test_epoch_end(self, outputs):
        print("[train] End Of Test")
        self.metric_fn(outputs, prefix='test')
        pass

    def predict_step(self, batched_data, batch_idx):
        x, id = batched_data
        y_hat, loss = self(x)  # forward
        return [y_hat, id]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'lr',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_lr_parameters(parser):
        # lr
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_updates', type=int, default=1000)
        parser.add_argument('--peak_lr', type=float, default=4e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        # Task
        parser.add_argument('--pose_sel', type=bool, default=False)
