import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange


def init_embedding(layer):
    if isinstance(layer, nn.Embedding):
        if len(layer.weight) > 1:
            layer.weight.data[0].zero_()

    if isinstance(layer, nn.Linear):
        layer.bias.data.zero_()


def init_layer(layer):
    # exception
    if isinstance(layer, RBFLayer):
        return
    for p in layer.parameters():
        if p.dim() > 1:
            # weight
            torch.nn.init.xavier_uniform_(p)
        else:
            # bias
            nn.init.constant_(p.data, 0.)
    # if it is embedding, set the first padding to be zero
    if isinstance(layer, nn.Embedding):
        layer.weight.data[0].zero_()


def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))


def init_last_residual_layer(layer):
    layer.weight.data.zero_()
    # torch.nn.init.xavier_uniform_(layer.weight.data)
    layer.bias.data.zero_()


def get_attn_bias(D, ndata):
    B, max_N = ndata.size()[:2]
    attn_bias = torch.zeros([B, max_N + 1, max_N + 1], dtype=torch.float32).to(D)
    # too_far_away = ((D > attention_dist_cutoff) | (D == -1.))
    too_far_away = D == -1.  # for padding only now
    attn_bias[:, 1:, 1:][too_far_away] = float('-inf')
    # VN
    pad_matrix = ndata[:, :, 0] == 0  # focus on both ligand and pocket
    attn_bias[:, 0, 1:][pad_matrix] = float('-inf')
    return attn_bias


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, num_layer):
        super(MultiHeadAttention, self).__init__()

        self.num_layer = num_layer
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        # QKV
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        # edge
        self.linear_e = nn.Linear(hidden_size, num_heads * att_size)
        self.e_output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.init_param()

    def init_param(self):
        init_layer(self.linear_q)
        init_layer(self.linear_k)
        init_layer(self.linear_v)
        init_last_residual_layer(self.output_layer)
        #
        init_layer(self.linear_e)
        init_last_residual_layer(self.e_output_layer)

    def forward(self, q, k, v, e, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        e = rearrange(self.linear_e(e), 'b i j (h k) -> b h i j k', h=self.num_heads, k=d_k)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        # w = torch.einsum('b h i d, b h k d -> b h i k', q, k)
        # w = w + e
        qk_e = torch.einsum('b h i d, b h j d, b h i j d -> b h i j d', q, k, e)  # too damn slow?
        w = qk_e.sum(dim=-1)
        ##
        # Attention-Core
        if attn_bias is not None:
            w_bias = w + attn_bias
        else:
            w_bias = w

        w_softmax = torch.softmax(w_bias, dim=3)
        ###
        x = self.att_dropout(w_softmax)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size

        # e_output
        w = self.att_dropout(rearrange(qk_e, 'b h i j d -> b i j (h d)'))
        w = self.e_output_layer(w)
        return x, w


class Interaction_Layer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, num_layer):
        super(Interaction_Layer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, num_layer)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        # self.post_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y, qk = self.self_attention(y, y, y, attn_bias, None, True)
        new_qk = qk[:, :, 1:, 1:]  # B, num_heads, l, l
        new_qk = torch.where(torch.isinf(new_qk), torch.zeros_like(new_qk), new_qk)
        new_qk = new_qk.permute(0, 2, 3, 1)
        #
        # new_qk = self.post_norm(new_qk)
        return new_qk


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        #
        self.cut_off = 10.

    def cutoff_fn(self, D):
        x = D / self.cut_off
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(x))

    def forward(self, x, edge_types):
        # Affine by pair-type
        # mul = self.mul(edge_types)
        # bias = self.bias(edge_types)
        # x = mul * x.unsqueeze(-1) + bias
        ###
        # GBF
        x = x.unsqueeze(-1)
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        # mask x, avoid padding error
        # padding_mask = (~(x == -1)).float()
        # x = x * padding_mask
        #
        gaussian_output = self.cutoff_fn(x) * gaussian(x.float(), mean, std).type_as(self.means.weight)
        return gaussian_output


# This one is modified ones, which is much better than the original gaussian kernels function.... about 0.05 R
class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10., use_affine_linear=False, edge_types=29 ** 2, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))

        # pair-type
        self.use_affine_linear = use_affine_linear
        # if use_affine_linear:
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        # if it exceeds the cut_off, value will be zero (make no penalty to edge feature)
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, torch.zeros_like(x))

    def forward(self, D, edge_types):
        # D=[b, n, n]->[b, n, n, 1]
        D = D.unsqueeze(-1)
        if self.use_affine_linear:
            # with affine
            mul = self.mul(edge_types)
            bias = self.bias(edge_types)
            D = mul * D + bias
        # with cut_off_fn
        rbf_D = self.cutoff_fn(D) * torch.exp(-self.widths * torch.pow((torch.exp(-D) - self.centers), 2))
        return rbf_D


class ComplexEncoder(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim, num_heads, intput_dropout_rate, angle_feat_size,
                 K=128,
                 rbf_cutoff=10., use_affine_linear=False, num_atom_types=29):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        ###
        # Node Feat
        self.atom_encoder = nn.Embedding(512 * node_feat_size + 1, hidden_dim, padding_idx=0)
        ####
        # Angle feat
        # 0=degree, 1=avg_bond_angles
        # self.angle_encoder = nn.Linear(1, hidden_dim)
        # # Degree Feat
        # self.degree_encoder = nn.Embedding(
        #   512, hidden_dim, padding_idx=0)
        ##
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        # virtual Node
        self.graph_token = nn.Embedding(1, hidden_dim)
        ####
        # Edge feat
        if edge_feat_size > 0:
            self.edge_encoder = nn.Embedding(512 * edge_feat_size + 1, hidden_dim, padding_idx=0)
        # Pair Affine
        self.num_atom_types = num_atom_types
        # Gaussian Kernels
        self.rbf = RBFLayer(K, cutoff=rbf_cutoff, use_affine_linear=use_affine_linear,
                            edge_types=self.num_atom_types ** 2)
        self.rel_pos_3d_proj = nn.Linear(K, hidden_dim)
        ###
        self.init_param()
        pass

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def atom_type2pair_type(self, x: torch.Tensor) -> torch.Tensor:
        atoms_type_x = x % self.num_atom_types  # 29-th is the 0-idx in pocket-atoms-type
        pair_type = atoms_type_x[:, :, None, :] * self.num_atom_types + atoms_type_x[:, None, :]
        return pair_type

    def wrap_bias(self, attn_bias):
        attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        return attn_bias

    def edge_feat(self, D, batched_data):
        x = batched_data['x']
        # Distance with RBF layer, convert 3d pos to an unlimited space
        D = D.float()
        # Pair-Type-Aware distance
        D, pair_type = D.squeeze(-1), self.atom_type2pair_type(x).squeeze(-1)
        real_3d_input = self.rel_pos_3d_proj(self.rbf(D, pair_type))
        edge_feats = real_3d_input
        ####
        # Edge feature[PLIP]
        if hasattr(self, 'edge_encoder'):
            edata = batched_data['edata']
            edge_input = self.edge_encoder(
                edata).sum(dim=-2)
            edge_feats = edge_feats + edge_input
        ###
        # Edge-feat
        # [n_graph, n_node+1, n_node+1, d]
        graph_edge_feats = torch.zeros(
            [edge_feats.size(0), edge_feats.size(1) + 1, edge_feats.size(1) + 1, edge_feats.size(3)]).to(edge_feats)
        graph_edge_feats[:, 1:, 1:] = edge_feats
        return graph_edge_feats

    def forward(self, batched_data):
        # Only Output Node Feats
        x = batched_data['x']
        ######
        n_graph, n_node = x.size()[:2]
        ########
        # Node-feat
        node_feature = self.atom_encoder(x).sum(dim=-2)
        ###
        # Other-Node-Feat
        # if angle_feat is not None:
        #   degree, angle_feat = angle_feat[:, :, 0].long(), angle_feat[:, :, 1:]
        #   # Heavy-Degree
        #   node_feature = node_feature + self.degree_encoder(degree).squeeze(-2)
        #   # Bond-Angle
        #   # node_feature = node_feature + self.angle_encoder(angle_feat)

        # Virtual Node Embedding
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)
        graph_node_feature = self.input_dropout(
            graph_node_feature)  # normally it is 0., because it will hurt performance a lot
        ##
        return graph_node_feature


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        # self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

        self.init_param()

    def init_param(self):
        init_layer(self.layer1)
        init_layer(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.gelu(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, num_layer):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, num_layer)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        # Edge
        self.edge_attn_norm = nn.LayerNorm(hidden_size)
        self.edge_ffn_norm = nn.LayerNorm(hidden_size)
        self.edge_ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

        self.init_param()

    def init_param(self):
        init_last_residual_layer(self.ffn.layer2)
        init_last_residual_layer(self.edge_ffn.layer2)

    def forward(self, x, e, attn_bias=None):
        # 1. Self-Attention
        y = self.self_attention_norm(x)
        e_hat = self.edge_attn_norm(e)
        y, e_hat = self.self_attention(y, y, y, e_hat, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        # 2. edge feat
        e_hat = self.self_attention_dropout(e_hat)
        e = e + e_hat
        e_hat = self.edge_ffn_norm(e)
        e_hat = self.edge_ffn(e_hat)
        e_hat = self.ffn_dropout(e_hat)
        e = e + e_hat
        return x, e


class PositionwiseFeedForward(nn.Module):
    # "Implements FFN equation."

    def __init__(self, d_model, d_ff, activation="PReLU", dropout=0.1, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.PReLU()
        # self.act_func = nn.LeakyReLU()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.W_1.weight)
        nn.init.xavier_normal_(self.W_2.weight)

    def forward(self, x):
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))


class SE3Head(nn.Module):
    def __init__(self, num_heads):
        super(SE3Head, self).__init__()
        self.qk_linear = nn.Linear(num_heads, num_heads)
        self.qk2scalar = nn.Linear(num_heads, 1)

    def forward(self, x, qk):
        qk = qk.permute(0, 2, 3, 1)[:, 1:, 1:, :]  # exclude virtual node
        qk = torch.where(torch.isinf(qk), torch.zeros_like(qk), qk)
        qk = self.qk_linear(qk)
        c = self.qk2scalar(F.relu(qk))
        # new_coords
        # x=[b, n, 3]
        rel_pos = x[:, :, None, :] - x[:, None, :, :]
        x_new = torch.sum(rel_pos * c, dim=-2)
        x_hat = x + x_new
        return x_hat

    def contact(self, pair_emb, d, pair_mask):
        # We only consider inter-interaction for now
        contact_D = (d < self.dist_cut_off).float()
        # contact loss
        pair_emb = pair_emb * pair_mask
        contact_pred = self.ContactHead(pair_emb)
        pos = contact_D[pair_mask].sum()
        neg = pair_mask.sum() - pos
        bce_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=neg / pos)
        contact_loss = bce_fn(contact_pred, contact_D) * pair_mask
        contact_loss = contact_loss.mean()  # try it out first
        # output contact_pred
        self.contact_pred = torch.sigmoid(contact_pred) * pair_mask  # use it later
        return contact_loss

    # code saving
    def cal_vina_terms(self, D, vdw_pair, hydro_pair, hbond_pair):
        d = D - vdw_pair
        zero_tensor = torch.tensor(0.).to(d)
        one_tensor = torch.tensor(1.).to(d)
        #
        vdw0 = torch.exp(-4. * d * d)
        vdw1 = torch.exp(-0.25 * (d - 3.) * (d - 3.))
        vdw2 = torch.where(d < 0., d * d, zero_tensor)
        #
        hydro_mask = torch.where(d >= 1.5, zero_tensor, d)
        hydro = torch.where(d <= 0.5, one_tensor, 1.5 - d) * hydro_mask * hydro_pair
        #
        hbond_mask = torch.where(d >= 0., zero_tensor, d)
        hbond = torch.where(d <= 0.7, one_tensor, d * -1.4285) * hbond_mask * hbond_pair
        # output
        terms = [vdw0, vdw1, vdw2, hydro, hbond]
        terms = torch.cat(terms, dim=-1)
        return terms

    def vinascore(self, D, pair_type, pair_mask, ligand_mask, pair_emb, contact_pred):
        self.vina_terms_weights = torch.tensor([-.0356, -.00516, 0.84, -.0351, -.587]).view(1, 1, 1, 5)
        self.VinaHead.W_2.weight.data.zero_()
        self.vina_terms_learner = torch.nn.Parameter(self.vina_terms_weights)
        # TODO: DEBUG
        # close_pairs_mask = (contact_pred > 0.5).float()
        close_pairs_mask = (D < 8.).float()
        # get vdw
        vdw_pair = self.vdw_table(pair_type)  # [b, n, n, 1]
        # get hydrophobic
        hydro_pair = self.hydro_table(pair_type)
        # get hbond
        hbond_pair = self.hbond_table(pair_type)
        # cal vinascore
        vina_weights = self.vina_terms_weights.to(D)
        #
        terms = self.cal_vina_terms(D, vdw_pair, hydro_pair, hbond_pair)
        terms = (terms * vina_weights).sum(dim=-1, keepdims=True) * .5  # we calculated energy twice
        # inter_terms
        inter_terms = terms * close_pairs_mask * pair_mask
        inter_terms = inter_terms.view(inter_terms.size(0), -1).sum(dim=-1, keepdims=True)
        # intra_terms  # not accurate at all
        # intra_self_mask = ((D != 0.) & (D > 3.5))
        # intra_terms = terms * close_pairs_mask * ligand_mask * intra_self_mask
        # intra_terms = intra_terms.view(intra_terms.size(0), -1).sum(dim=-1, keepdims=True)
        # final vina score
        vina_score = inter_terms  # delta G
        # convert2pIC50
        # vina_score = vina_score.clip(-99., 0)
        # log_vina_score = -torch.log10(torch.exp(vina_score / .59))
        return vina_score
