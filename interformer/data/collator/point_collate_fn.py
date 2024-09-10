import torch


# for point-cloud data
def pointformer_collate_fn(data):
    names, X, Y, id = list(zip(*data))
    Y = torch.cat(Y, dim=-1).view(-1, 1)
    ndata, edata, _, D, _, _, angle, ligand_len, targets = list(zip(*X))
    return Y
