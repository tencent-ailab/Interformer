import torch
import torch.nn as nn
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, AUROC

from model.ppi_model.ppi_model import PPI
from model.transformer.graphormer.graphformer_utils import init_embedding, ComplexEncoder, EncoderLayer, \
    PositionwiseFeedForward


class PPIScorer(PPI):
    def __init__(
        self,
        args
    ):
        super().__init__(args)
        hidden_dim = args['hidden_dim']
        num_heads = args['num_heads']
        attention_dropout_rate = args['attention_dropout_rate']
        dropout_rate = args['dropout_rate']
        intput_dropout_rate = args['intput_dropout_rate']
        ffn_dim = hidden_dim * args['ffn_scale']
        n_layers = args['n_layers']
        node_feat = args['node_feat_size']
        edge_feat = args['edge_feat_size']
        angle_feat_size = args['angle_feat_size']
        # RBF
        K = args['rbf_K']
        rbf_cutoff = args['rbf_cutoff']
        ###
        # Embedding
        self.complex_feat_layer = ComplexEncoder(node_feat, edge_feat, hidden_dim, num_heads, intput_dropout_rate,
                                                 angle_feat_size, K, rbf_cutoff, num_atom_types=22)
        ###
        # Intra
        self.intra_encoder = nn.ModuleList(
            [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
             for i in range(n_layers // 2)])
        # Inter
        self.inter_encoder = nn.ModuleList(
            [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
             for i in range(n_layers)])
        ####
        # Task-Layers
        final_dim = hidden_dim
        # Binary task
        self.final_ln = nn.LayerNorm(final_dim)
        self.out_proj = PositionwiseFeedForward(final_dim, ffn_dim, d_out=1, dropout=dropout_rate)
        ##

    def loss_fn(self, y_hat, batched_data):
        # 1. BCE for distinguish decoys and native poses
        y = batched_data['y']  # 0=decoy, 1=native
        if y.size(0):
            bce_loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = bce_loss_fn(y_hat, y)
        else:
            loss = 0.
        return loss

    def metric_fn(self, outputs, prefix='train'):
        y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        y = torch.cat([x['x']['y'] for x in outputs], dim=0).int()
        log_name = prefix + '_auroc'
        auroc_fn = AUROC(task='binary')
        auroc_score = auroc_fn(y_hat, y)
        self.log(log_name, auroc_score, sync_dist=True)

    def forward(self, batched_data):
        node_feats = self.complex_feat_layer(batched_data)  # [b, n, d]
        edge_feats = self.complex_feat_layer.edge_feat(batched_data['D'], batched_data)
        ###
        # intra
        # intra_mask, [n_graph, n_heads, n_node+1, n_node+1]
        intra_mask = self.complex_feat_layer.wrap_bias(batched_data['intra_mask'])
        for i in range(len(self.intra_encoder)):
            node_feats, _ = self.intra_encoder[i](node_feats, edge_feats, intra_mask)
        ###
        # inter
        # inter_mask, [n_graph, n_heads, n_node+1, n_node+1]
        inter_mask = self.complex_feat_layer.wrap_bias(batched_data['inter_mask'])
        for i in range(len(self.inter_encoder)):
            node_feats, _ = self.inter_encoder[i](node_feats, edge_feats, inter_mask)
        ####
        # Pooling
        output_node = node_feats
        vn_node = self.final_ln(output_node[:, 0, :])
        y_hat = self.out_proj(vn_node)
        ##
        # Loss
        loss = self.loss_fn(y_hat, batched_data)
        return y_hat, loss

    @staticmethod
    def add_model_specific_args(args):
        parser = args.add_argument_group("PPIScorer")
        # dim
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--ffn_scale', type=int, default=4)
        # drop
        parser.add_argument('--intput_dropout_rate', type=float, default=0.)
        parser.add_argument('--dropout_rate', type=float, default=0.2)
        parser.add_argument('--attention_dropout_rate', type=float, default=0.2)
        # RBF
        parser.add_argument('--rbf_K', type=int, default=128)
        parser.add_argument('--rbf_cutoff', type=float, default=10.)
        # common used parameters
        PPI.add_lr_parameters(parser)
        return args


if __name__ == '__main__':
    from utils.parser import get_args
    from data.data_process import GraphDataModule

    args = get_args()
    args['debug'] = True
    args['inference'] = True
    # args['pocket_path'] = './'
    # args['data_path'] = f"{args['work_path']}/{args['pocket_path']}/sabdab_dataset.csv"
    # DataLoader
    dm = GraphDataModule(args, istrain=False)
    dm.setup()
    dataset = dm.bind_test
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dm.collate_fn)
    # model
    model = PPIScorer(args)
    ###
    # Run
    batch_id = 0
    for items in dataloader:
        print(batch_id)
        batch_id += 1
        #
        X, _ = items
        y_hat = model(X)
    print("Test DONE.")
