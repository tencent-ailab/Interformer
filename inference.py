import glob
import os
import shutil

# DDP inference
import pytorch_lightning as pl
import pickle
import torch.distributed as dist
from collections import defaultdict
from torchmetrics.functional import pearson_corrcoef
import torch
import numpy as np
#
from utils.parser import get_args
from utils.train_utils import load_from_checkpoint, param_count, get_checkpoint_realpath, refresh_args
from data.dataset.bindingdata import BindingData
from data.data_process import data_loader_warper
from utils.eval import cal_per_target

os.environ["NCCL_IB_DISABLE"] = "1"
args = get_args()
args['inference'] = True
args['device'] = torch.device(args['gpus'])
os.makedirs('result', exist_ok=True)
# rm energy folder's files
if args['energy_output_folder']:
    energy_output_folder = args['energy_output_folder'] + '/gaussian_predict'
    if os.path.exists(energy_output_folder):
        print(f'rm {energy_output_folder}')
        shutil.rmtree(energy_output_folder)
    os.makedirs(energy_output_folder, exist_ok=True)


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
        N = ordered_results[0][1].size(0)
        pred, Y, targets, idx = [], [], [], []
        for batch_data in ordered_results:
            pred.append(batch_data[0])
            Y.append(batch_data[1].cpu())
            targets.extend(batch_data[2])
            idx += list(batch_data[3])
        # gathering
        Y = torch.cat(Y)
        targets = np.array(targets)
        # multi-task
        num_hat = len(ordered_results[0][0])
        all_pred = []
        for i in range(num_hat):
            one_hat = [x[i].cpu() for x in pred]
            # Loss Output, skip
            if len(one_hat[0].shape) == 0:
                continue
            all_pred.append(torch.cat(one_hat))
        all_pred = torch.cat(all_pred, dim=-1)
        # reorder by idx
        real_idx = [x[1] for x in sorted([(id, i) for i, id in enumerate(idx)])]
        Y = Y[real_idx]
        all_pred = all_pred[real_idx]
        targets = targets[real_idx].tolist()
        return all_pred, Y, targets


dfs = {}
ensemble_results = defaultdict(list)
models_N = 0
rank = 0
models_names = []
for checkpoint_folder in args['ensemble']:
    checkpoint = get_checkpoint_realpath(checkpoint_folder, posfix=args['posfix'])
    if checkpoint == '':
        print(f"# Model doesn't Exists:{checkpoint_folder}")
        continue

    # record
    models_N += 1
    models_names.append(checkpoint[:checkpoint.rfind('/check')])
    #
    model = load_from_checkpoint(args, checkpoint).eval()
    args = refresh_args(args, model)
    model_args = model.hparams.args
    param_count(model, print_model=False)
    inferencer = pl.Trainer(devices=args['gpus'], accelerator='cuda', strategy='ddp', precision=args['precision'],
                            logger=False)
    for csv in args['test_csv']:
        print("# csv <--", csv)
        args['data_path'] = csv
        # dataset
        bind_data = BindingData(args, n_jobs=1 if args['debug'] else args['n_jobs'], istrain=False)
        df, dataset = bind_data.df, bind_data.datasets
        loader = data_loader_warper(dataset, args['batch_size'],
                                    energy_mode=args['energy_mode'] if 'energy_mode' in args else False,
                                    model=args['model'],
                                    num_workers=0 if args['debug'] else 5)
        # predict
        res = inferencer.predict(model, dataloaders=loader)
        results = collect_results_gpu(res)
        rank = dist.get_rank()
        if rank == 0:
            #########
            # calculate score
            pred, Y, targets = results
            pred = pred.float()
            if pred.size(0) and not model_args['energy_mode']:
                # multi-task output
                if pred.size(-1) > 1:
                    affinity_pred, pose_pred = pred[:, 0].view(-1, 1), torch.sigmoid(pred[:, 1])
                else:
                    affinity_pred = pred
                ##
                msg, per_target_mean_score = cal_per_target(affinity_pred, Y, targets, pearson_corrcoef)
                score_msg = f"{msg} <- {checkpoint} <- {csv}"
                dfs[csv] = bind_data.df
                # write into csv
                folder_base_name = checkpoint_folder.split('/')[-2] + '_' + checkpoint_folder.split('/')[-1]
                csv_name = os.path.basename(csv)
                bind_data.df['pred_pIC50'] = affinity_pred.numpy()
                if pred.size(-1) > 1:
                    bind_data.df['pred_pose'] = pose_pred.numpy()
                bind_data.df.to_csv(f'result/{folder_base_name}_{csv_name}', index=False)
                # records
                saver = [score_msg, affinity_pred, per_target_mean_score]
                if pred.size(-1) > 1:
                    saver.append(pose_pred)
                ensemble_results[csv].append(saver)
    print('%' * 100)


def ensemble(ensemble_results):
    print('=' * 100)
    for k, items in ensemble_results.items():
        msgs = [x[0] for x in items]
        print('\n'.join(msgs))
        print('')

    ########
    print("@" * 100 + f"\n# Selected Model index..")
    # Ensemble part
    TOP = 8
    # select the top 5 models to do ensemble
    models_scores = [[] for _ in range(models_N)]
    individual_records = [[] for _ in range(models_N)]
    for csv, items in ensemble_results.items():
        for i in range(len(items)):
            target_mean_score = items[i][2]
            csv_base_name = os.path.basename(csv).split('.')[0]
            models_scores[i].append(target_mean_score)  # name, score
            individual_records[i].append(f"{csv_base_name}:{target_mean_score:.2f}")
    mean_scores = [np.mean(x) for x in models_scores]
    selected_indices = np.argsort(mean_scores)[::-1][:TOP].tolist()
    #
    selected_models_info = [[models_names[i], f'{mean_scores[i]:.6f}', ','.join(individual_records[i])] for i in
                            selected_indices]
    for model_info in selected_models_info:
        print(' -> '.join(model_info))
    #
    print('=' * 100)
    print("# Calculating Ensemble results....")
    for csv_name, items in ensemble_results.items():
        base_name = os.path.basename(csv_name)[:-4]
        out_name = f'result/{base_name}_ensemble.csv'
        # Ensemble
        csv = dfs[csv_name]
        # affinity
        pred_list = [x[1] for i, x in enumerate(items) if i in selected_indices]
        final_pic50 = torch.cat([x.view(1, -1) for x in pred_list], dim=0)
        csv['pred_pIC50'] = torch.mean(final_pic50, dim=0).numpy()
        csv['pred_pIC50_var'] = torch.var(final_pic50, dim=0).numpy()
        # Pred-pose
        if len(items[0]) > 3:
            pred_list = [x[3] for i, x in enumerate(items) if i in selected_indices]
            final_pose = torch.cat([x.view(1, -1) for x in pred_list], dim=0)
            csv['pred_pose'] = torch.mean(final_pose, dim=0).numpy()
        # Output CSV files
        print(out_name)
        print(csv.corr('pearson', numeric_only=True)['pIC50'])
        csv.to_csv(out_name, index=False)
        print('\n' + '-' * 100)


##########
# Master Calculate
if rank == 0:
    ensemble(ensemble_results)
