import torch


def get_reg_loss_fn(args):
    mse_loss_fn = torch.nn.MSELoss(reduction='none')
    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0, reduction='none')

    def cal_ranking_loss(preds, targets):
        d = targets[:, None].repeat(1, len(targets))
        x1 = preds[:, None].repeat(1, len(targets))
        x2 = x1.transpose(0, 1)
        y = (d > d.transpose(0, 1)) * 2 - 1
        ranking_loss = ranking_loss_fn(x1, x2, y)
        return ranking_loss

    def cal_huber_loss(diff):
        delta = 6.
        # origin huber_loss
        # huber_loss = torch.where(torch.abs(diff) < delta, 0.5 * torch.pow(diff, 2.0),
        #                          delta * (torch.abs(diff) - 0.5 * delta))
        # pesudo-huber_loss
        div = (diff / delta)
        huber_loss = (delta ** 2) * (torch.sqrt(1. + div * div) - 1.)
        return huber_loss

    def cal_mse_loss(preds, targets):
        mse_loss = mse_loss_fn(preds, targets).float().mean()
        return mse_loss

    def reg_fn(preds, targets):
        # mse_loss = mse_loss_fn(preds, targets).float().mean()
        # rank_loss = cal_ranking_loss(preds, targets).float().mean()
        ###########
        # Neg + Pos-> hinge_loss
        # exclude_noaffinity_data
        no_affinity_masks = torch.where(torch.abs(targets) <= 0.1, 0., 1.).long()
        all_loss = torch.tensor(0.)
        targets, preds = targets[no_affinity_masks != 0], preds[no_affinity_masks != 0]
        if len(targets):
            diff = torch.where(targets > 0., preds - targets, torch.maximum(preds + targets, torch.zeros_like(preds)))
            huber_loss = cal_huber_loss(diff).float()
            # rebalance the positive samples's loss
            label_weights = torch.where(targets > 0., torch.tensor(1.).to(huber_loss),
                                        torch.tensor(args['neg_weight']).to(huber_loss))
            huber_loss = huber_loss * label_weights
            huber_loss = huber_loss.mean()
            # all_loss = rank_loss + huber_loss
            all_loss = huber_loss
        return all_loss

    return reg_fn


def get_dist_loss_fn():
    mse_loss_fn = torch.nn.MSELoss(reduction='none')

    def debug(targets):
        top_targets = targets[0, :, :10]
        cloest_amino = torch.where(top_targets > 0, top_targets, torch.tensor(9999.).to(targets)).min(0)
        dist_str = [f'{x:.4f}' for x in cloest_amino[0].view(-1).tolist()]
        return [dist_str, '@', cloest_amino[1].view(-1).tolist()]

    def reg_fn(preds, targets):
        mask = (~(targets == -1.)).float()
        loss = mse_loss_fn(preds, targets) * mask
        loss = loss.mean()
        # debug
        # print(debug(preds), '||||', debug(targets), loss)
        return loss

    return reg_fn


def get_pose_sel_loss_fn():
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    def cls_fn(preds, targets):
        targets = torch.where(targets >= 0, 1., 0.)
        loss = bce_loss_fn(preds, targets)
        loss = loss.mean()
        return loss

    return cls_fn
