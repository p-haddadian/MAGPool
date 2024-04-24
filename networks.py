import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from layers import MAGPool


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden = args.hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.hops = args.hop_num
        self.alpha = args.alpha

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.pool1 = MAGPool(self.hidden, self.pooling_ratio, self.hops, self.alpha, dropout_ratio= self.dropout_ratio)

        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.pool2 = MAGPool(self.hidden, self.pooling_ratio, self.hops, self.alpha, dropout_ratio= self.dropout_ratio)

        self.conv3 = GCNConv(self.hidden, self.hidden)
        self.pool3 = MAGPool(self.hidden, self.pooling_ratio, self.hops, self.alpha, dropout_ratio= self.dropout_ratio)

        self.conv4 = GCNConv(self.hidden, self.hidden)
        self.pool4 = MAGPool(self.hidden, self.pooling_ratio, self.hops, self.alpha, dropout_ratio= self.dropout_ratio)

        self.lin1 = Linear(self.hidden * 2, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)
        
    def forward(self, data):
        ###roya
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    