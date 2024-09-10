from data.collator.inter_collate_fn import unpack_dict, pad_1D, get_attn_bias
from constant import *
import torch


def ppi_collate_fn(batch):
    names, X, id = list(zip(*batch))
    X = unpack_dict(X)
    # ndata
    ndata = torch.nn.utils.rnn.pad_sequence(X['ndata'], batch_first=True)  # [b, n, 1]
    B, max_N, _ = ndata.size()
    # xyz
    xyz = torch.cat([pad_1D(x, max_N) for x in X['xyz']]).float()
    D_mask = (ndata == 0.).permute(0, 2, 1).repeat(1, max_N, 1)
    D = torch.cdist(xyz, xyz)
    D = torch.where(D_mask, torch.tensor(-1.), D)
    # attn_bias
    attn_bias = get_attn_bias(D, ndata)  # [b, n + 1, n + 1]
    D = D.unsqueeze(-1)  # [b, n, n, 1]
    ###
    # Label
    y = torch.tensor(X['isnative']).view(-1, 1).float()
    ####
    # Over Max-atoms, max_ppi_complex_nodes=320
    if ndata.shape[1] > max_ppi_complex_nodes:
        # print(f"# prune.{ndata.shape}")  # Debug
        ndata = ndata[:, :max_ppi_complex_nodes]
        D = D[:, :max_ppi_complex_nodes, :max_ppi_complex_nodes]
        attn_bias = attn_bias[:, :max_ppi_complex_nodes + 1, :max_ppi_complex_nodes + 1]
    ##
    # packing
    #
    X_new = {'x': ndata, 'D': D, 'attn_bias': attn_bias, 'edata': None, 'y': y}
    return X_new, id


def _get_mask(attn_bias, D, lens, cut_off=10.):
    intra_mask = torch.ones_like(attn_bias) * -1.
    # avoid nan, at least there is one node to be attend
    intra_mask[:, range(intra_mask.size(1)), range(intra_mask.size(1))] = 0.
    # intra_mask[:, :, 0] = 0.
    inter_mask = intra_mask.clone()
    for i, row in enumerate(lens):
        h_len, l_len, ag_len = row
        intra_mask[i, 1: 1 + h_len, 1: 1 + h_len] = 0.  # heavy&heavy
        intra_mask[i, 1 + h_len: 1 + h_len + l_len, 1 + h_len: 1 + h_len + l_len] = 0.  # light&light
        intra_mask[i, 1 + h_len + l_len: 1 + h_len + l_len + ag_len,
        1 + h_len + l_len: 1 + h_len + l_len + ag_len] = 0.  # ag&ag
        ###
        # inter_mask
        inter_mask[i, 1: 1 + h_len + l_len, 1 + h_len + l_len: 1 + h_len + l_len + ag_len] = 0.  # heavy+light&ag
        inter_mask[i, 1 + h_len + l_len: 1 + h_len + l_len + ag_len, : 1 + h_len + l_len] = 0.  # ag&heavy+light
        # closest ag_indices
        ag_indices = (D[i, :h_len + l_len, h_len + l_len:h_len + l_len + ag_len].min(0)[0] < cut_off).view(-1)
        inter_mask[i, 0, :1 + h_len + l_len] = 0.  # vn&heavy+light
        inter_mask[i, 0, 1 + h_len + l_len: 1 + h_len + l_len + ag_len][ag_indices] = 0.  # vn&closest_ag
    # long-distance pair should not have attention
    intra_mask[:, 1:, 1:][D.squeeze(-1) > cut_off] = -1.
    inter_mask[:, 1:, 1:][D.squeeze(-1) > cut_off] = -1.
    # assign -inf
    intra_mask[intra_mask == -1] = float('-inf')
    inter_mask[inter_mask == -1] = float('-inf')
    return intra_mask, inter_mask


def ppi_residue_collate_fn(batch):
    names, X, id = list(zip(*batch))
    X = unpack_dict(X)
    # ndata
    ndata = torch.nn.utils.rnn.pad_sequence(X['ndata'], batch_first=True)  # [b, n, 6]
    f_ndata = ndata[:, :, 0].unsqueeze(-1)
    B, max_N, _ = ndata.size()
    # xyz
    xyz = torch.cat([pad_1D(x, max_N) for x in X['xyz']]).float()
    D_mask = (f_ndata == 0).permute(0, 2, 1).repeat(1, max_N, 1)
    D = torch.cdist(xyz, xyz)
    D = torch.where(D_mask, torch.tensor(-1.), D)
    ###
    # attn_bias
    attn_bias = get_attn_bias(D, f_ndata)  # [b, n + 1, n + 1]
    D = D.unsqueeze(-1)  # [b, n, n, 1]
    ###
    # Inter,Out-edge mask
    intra_mask, inter_mask = _get_mask(attn_bias, D, X['res_lens'])
    ###
    # Label
    y = torch.tensor(X['isnative']).view(-1, 1).float()
    ##
    # if ndata.shape[1] > max_ppi_complex_nodes:
    #   ndata = ndata[:, :max_ppi_complex_nodes]
    #   D = D[:, :max_ppi_complex_nodes, :max_ppi_complex_nodes]
    #   attn_bias = attn_bias[:, :max_ppi_complex_nodes + 1, :max_ppi_complex_nodes + 1]
    ####
    # packing
    #
    X_new = {'x': ndata, 'D': D, 'attn_bias': attn_bias, 'edata': None, 'y': y, 'inter_mask': inter_mask,
             'intra_mask': intra_mask}
    return X_new, id
