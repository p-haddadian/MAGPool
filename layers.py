import torch
from torch.nn import Dropout, LayerNorm
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn.pool.topk_pool import filter_adj
###roya
from torch_geometric.nn import Linear
from utils import sparse_adj

class MAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio = 0.5, hop_number = 3, alpha = 0.4, Conv=GATConv, non_linearity = torch.tanh, dropout_ratio = 0.3):
        super(MAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.hop_num = hop_number
        self.attention_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.alpha = alpha
        ###roya
        self.scorelayer=Linear(in_channels,1)
        self.feat_drop = Dropout(dropout_ratio)
        ### Parsa
        # self.graph_norm = LayerNorm(in_channels)  # Feature normalization

    def forward(self, x, edge_index, edge_attr = None, batch = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        scc, attention = self.attention_layer(x, edge_index, return_attention_weights = True)

        # scores = self.ppr_approximation(x, attention).squeeze()
        ### Parsa (LayerNormalization)
        # x = self.graph_norm(x)
        ### Roya
        sparse_attention = sparse_adj(attention[0], attention[1], x.size(0), 'sum')
        A_prime = self.ppr_approximation(x, sparse_attention)
        scores = self.scorelayer(A_prime)
        # scores = torch.matmul(A_prime, scores)
        scores = scores.reshape(-1)
        perm = topk(scores, self.ratio, batch)
        x = x[perm] * self.non_linearity(scores[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=scores.size(0))

        return x, edge_index, edge_attr, batch, perm

    def ppr_approximation(self, x, attention_scores):
        feat_0 = x
        feat = feat_0
        attentions = attention_scores
        for _ in range(self.hop_num):
            feat = torch.matmul(attentions, feat)
            feat = (1.0 - self.alpha) * feat + self.alpha * feat_0

            ###roya
            feat = self.feat_drop(feat)
        return feat
        