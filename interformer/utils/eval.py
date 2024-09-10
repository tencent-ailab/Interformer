import pickle

import numpy as np
import torch
import torch.distributed as dist
from torchmetrics import AUROC


def cal_per_target(pred, Y, targets, evaluator):
    pred, Y = pred.view(-1), Y.view(-1)
    targets = np.array(targets)
    whole_score = evaluator(pred, Y)
    uni_targets = np.unique(targets)
    scores = []
    for t in uni_targets:
        pred_t = pred[targets == t]
        Y_t = Y[targets == t]
        score = evaluator(pred_t, Y_t)
        if len(Y_t) < 10 or np.isnan(score):
            continue
        scores += [score]
    final_score = np.mean(scores)
    #
    msg = f"R:{whole_score: .6f}/Per-Target-R:{final_score: .6f}"
    return msg, final_score


def collect_results_gpu(result_part):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    torch.cuda.synchronize()
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for single_gpu_res in part_list:
            for res in single_gpu_res:
                ordered_results.extend([list(res)])
        ###########
        # merge gpus result together
        pred, idx = [], []
        for batch_data in ordered_results:
            pred.append(batch_data[0])
            idx += list(batch_data[-1])
        # multi-task
        num_class = ordered_results[0][0].size(1)  # first entry's [n, num_class]
        all_pred = []
        for i in range(num_class):
            one_hat = [x[:, i].cpu().view(-1, 1) for x in pred]
            # Loss Output, skip
            if len(one_hat[0].shape) == 0:
                continue
            all_pred.append(torch.cat(one_hat, dim=0))
        all_pred = torch.cat(all_pred, dim=-1)
        # reorder by idx
        real_idx = [x[1] for x in sorted([(id, i) for i, id in enumerate(idx)])]
        all_pred = all_pred[real_idx]
        return all_pred


def cal_auroc(y_hat, y):
    auroc_fn = AUROC(task='binary')
    auroc = auroc_fn(y_hat, y)
    msg = f'AUROC={auroc}'
    return msg, auroc
