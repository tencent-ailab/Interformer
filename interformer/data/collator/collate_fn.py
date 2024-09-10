import torch
import numpy as np
from constant import *


# [n, ?]
def pad_1D(x, padlen, pad_value=0.):
    xlen, d = x.size(0), x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, d], dtype=x.dtype).fill_(float(pad_value))
        new_x[:xlen, :d] = x
        x = new_x
    return x.unsqueeze(0)


# [n, n]
def pad_2D(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float(-1.))
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


# [s, r, f]
def pad_pair(X, f_size=None, fill_val=-1.):
    x1_pad_len = max([x.size(0) for x in X])
    x2_pad_len = max([x.size(1) for x in X])
    f_size = f_size if f_size else X[0].size(-1)
    #
    res = []
    for i, x in enumerate(X):
        x1len, x2len = x.size(0), x.size(1)
        new_x = x.new_zeros([x1_pad_len, x2_pad_len, f_size], dtype=x.dtype).fill_(fill_val)
        new_x[:x1len, :x2len, :] = x
        res += [new_x.unsqueeze(0)]
    res = torch.cat(res)
    return res


# [n, n, f]
def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_pair_mask(D, ligand_len):
    pair_mask = torch.ones_like(D)
    # <7.5A connection
    for i, l in enumerate(ligand_len):
        pair_mask[i, :l, :l] = 0.
        pair_mask[i, l:, l:] = 0.
    ###
    # convert into bool
    pair_mask = torch.where(D == -1., torch.tensor(0.).to(pair_mask), pair_mask).bool()  # [b, n, n, 1]
    return pair_mask


def get_attn_bias(D, ndata, dist_cut_off=True):
    B, max_N = ndata.size()[:2]
    attn_bias = torch.zeros([B, max_N + 1, max_N + 1], dtype=torch.float32).to(D)
    ##
    if dist_cut_off:
        too_far_away = ((D > attention_dist_cutoff) | (D == -1.))
    else:
        too_far_away = D == -1.  # for padding only now
    attn_bias[:, 1:, 1:][too_far_away] = float('-inf')
    # VN
    pad_matrix = ndata[:, :, 0] == 0  # focus on both ligand and pocket
    attn_bias[:, 0, 1:][pad_matrix] = float('-inf')
    return attn_bias
