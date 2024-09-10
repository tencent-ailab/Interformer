import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
from model.model_utils import PolynomialDecayLR
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, AUROC
from model.loss_utils import get_reg_loss_fn, get_pose_sel_loss_fn
from model.mixup import mixup_losses

from torchmetrics import MeanMetric


class SBDD(pl.LightningModule):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.save_hyperparameters()
        # task
        self.energy_mode = args['energy_mode']
        self.pose_sel_mode = args['pose_sel_mode']
        # loss
        self.loss_fn = []
        self.all_evaluations = []
        self.add_affinity_eval_log(args)
        # learning scheduler
        self.warmup_updates = args['warmup_updates']
        self.tot_updates = args['warmup_updates'] * 15  # for simplicity
        self.peak_lr = args['peak_lr']
        self.end_lr = args['end_lr']
        self.weight_decay = args['weight_decay']
        self.clip = args['clip']
        # log
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def add_affinity_eval_log(self, args):
        # reg
        print("# [Eval] Using Affinity Reg Task.")

        def affinity_eval_fn(y_hat, y_gt, metric, log_name):
            affinity_mask = (abs(y_gt) > 0.1).bool()
            y_gt = y_gt[affinity_mask]
            y_hat = y_hat[affinity_mask].float()
            # Positive Only
            mask = (y_gt > 0.1).bool()
            y_hat_pos = y_hat[mask]
            y_gt_pos = y_gt[mask]
            if len(y_gt_pos):
                metric(y_hat_pos, y_gt_pos)

        self.add_metric('affinity', loss_fn=get_reg_loss_fn(args),
                        metric=PearsonCorrCoef(), eval_fn=affinity_eval_fn)
        # cls
        if self.pose_sel_mode:
            print("# [Eval] Using Pose Selection CLS task.")

            def affinity_cls_eval_fn(y_hat, y_gt, metric, log_name):
                cls_y_gt = torch.where(y_gt >= 0., torch.tensor(1.).to(y_gt), torch.tensor(0.).to(y_gt)).long()
                metric(y_hat, cls_y_gt)

            self.add_metric('pose_selection', loss_fn=get_pose_sel_loss_fn(),
                            metric=AUROC(task='binary'),
                            eval_fn=affinity_cls_eval_fn)
            # forcing reg task to be 0.1
            self.loss_fn[0][1] = 0.1

    def add_metric(self, name, weight=1., loss_fn=None, metric=None, eval_fn=None):
        if metric is None:
            metric = MeanMetric()
        self.loss_fn.append([loss_fn, weight])
        self.all_evaluations.append([name, eval_fn])
        self.__setattr__(name, metric)

    def debug_info(self, batched_data, batch_idx):
        x, y_gt, _, _ = batched_data
        num_pos, num_neg = (y_gt > 0.).sum(), (y_gt < 0.).sum()
        print(f"Pos:{num_pos}, Neg:{num_neg}, idx:{batch_idx}")

    def eval_main(self, outputs, prefix='train'):
        # grep metric first
        metrics = []
        for (eval_name, evan_fn) in self.all_evaluations:
            metrics.append(self.__getattr__(eval_name))
        # metric calculate together
        total = 0
        for j, (item) in enumerate(outputs):
            y_hat, y_gt = item['y_hat'], item['y_gt']
            total += len(y_gt)
            for i, (eval_name, eval_fn) in enumerate(self.all_evaluations):
                if eval_fn:
                    eval_fn(y_hat[i], y_gt, metrics[i], '')
                else:
                    name, _ = self.all_evaluations[i]
                    self.log(f'{prefix}_{name}', y_hat[i], sync_dist=True)
        # final log
        for i, (eval_name, eval_fn) in enumerate(self.all_evaluations):
            if eval_fn:
                self.log(f'{prefix}_{eval_name}', metrics[i].compute(), sync_dist=True)
                metrics[i].reset()
        # print(f"[train] epoch num of samples:{total}")

    def training_step(self, batched_data, batch_idx):
        x, y_gt, _, _ = batched_data
        # print('Debug', x['target'], y_gt)
        y_hat = self(x)
        # Loss
        loss = self.merge_losses(y_hat, y_gt)
        #####
        # Clip
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self.log('train_clip_norm', norm, sync_dist=False)
        #####
        # Logging
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, batch_size=len(y_gt))
        ###
        outputs = {
            'y_hat': y_hat,
            'y_gt': y_gt,
            'loss': loss
        }
        self.train_step_outputs.append(outputs)
        return loss

    def on_train_epoch_end(self):
        print(f"[train] At validation_step, global_step={self.global_step}")
        self.eval_main(self.train_step_outputs, 'train')
        self.train_step_outputs.clear()

    def validation_step(self, batched_data, batch_idx):
        x, y_gt, _, _ = batched_data
        y_hat = self(x, istrain=False)  # forward
        loss = self.merge_losses(y_hat, y_gt)
        ###
        # Metrics
        self.log('val_loss', loss, sync_dist=True, batch_size=len(y_gt))
        ###
        outputs = {
            'y_hat': y_hat,
            'y_gt': y_gt,
            'loss': loss
        }
        self.val_step_outputs.append(outputs)
        return loss

    def on_validation_epoch_end(self):
        print(f"[train] At validation_step, global_step={self.global_step}")
        self.eval_main(self.val_step_outputs, 'val')
        self.val_step_outputs.clear()

    def test_step(self, batched_data, batch_idx):
        x, y_gt, _, _ = batched_data
        y_hat = self(x, istrain=False)  # forward
        loss = self.merge_losses(y_hat, y_gt)
        ###
        # Metrics
        self.log('test_loss', loss, sync_dist=True)
        outputs = {
            'y_hat': y_hat,
            'y_gt': y_gt,
            'loss': loss
        }
        self.test_step_outputs.append(outputs)
        return loss

    def on_test_epoch_end(self):
        print(f"[test], global_step={self.global_step}")
        self.eval_main(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def _debug_gradient(self, loss):
        # print gradients
        loss.backward()
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    print(name, m, m.weight.grad.std())
        # Debug print gradients
        # self.debug_info(batched_data, batch_idx)

    def merge_losses(self, y_hat, y_gt):
        all_losses = []
        for i, (loss_fn, w) in enumerate(self.loss_fn):
            if loss_fn:
                l = loss_fn(y_hat[i], y_gt)
            else:
                l = y_hat[i]
            all_losses.append(l * w)
        loss = sum(all_losses)
        return loss

    def predict_step(self, batched_data, batch_idx):
        x, y_gt, target, id = batched_data
        y_hat = self(x, istrain=False)  # forward
        return [y_hat, y_gt, target, id]

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
        parser.add_argument('--pose_sel_mode', type=bool, default=False)  # affinity & pose-selection Mode
        parser.add_argument('--energy_mode', type=bool, default=False)  # Energy Mode
