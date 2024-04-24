import torch
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csr_tensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import degree, is_undirected
from torch import Tensor
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def sparse_adj(edge_index: Tensor, edge_weight: Tensor, n_node: int, aggr: str, format: str = 'coo'):
    '''
    Convert edge_index and edge_weight to the sparse adjacency matrix.
    Args:
        edge_index (Tensor): An edge index to be converted in the shape of (2, |E|)
        edge_weight (Tensor): A tensor containing the weight assigned to each of the edges in the shape of (|E|)
        n_node (int): Number of the nodes in the graph
        aggr (str): How sparse adjacency matrix is going to be normalized (mean, sum, GCN)
    '''
    if n_node == -1:
        n_node = int(edge_index.max().item() + 1)
    
    # Convert edge weight to the form of a vector
    edge_weight = edge_weight.view(-1)

    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0   # preprocessing for isolated nodes
    ret = None
    if aggr == 'mean':
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == 'sum':
        val = edge_weight
    elif aggr == 'GCN':
        deg = torch.pow(deg, -0.5)
        val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
    else:
        raise ValueError('not defined aggregation function')

    if format == 'coo':
        ret = to_torch_coo_tensor(edge_index, val, (n_node, n_node))
    elif format == 'csr':
        ret = to_torch_csr_tensor(edge_index, val, (n_node, n_node))
    else:
        ret = SparseTensor(row= edge_index[0],
                        col= edge_index[1],
                        value= val,
                        sparse_sizes=(n_node, n_node)).coalesce()
    ret = ret.cuda() if edge_index.is_cuda else ret
    return ret


def plotter(losses, accuracies = None):
    plt.plot(losses[0], label='training loss')
    plt.plot(losses[1], label='validation loss')
    plt.title('Evaluating Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('plot-loss.png')
    plt.show()

    if accuracies != None:
        plt.plot(accuracies[0], label='training acc')
        plt.plot(accuracies[1], label='validation acc')
        plt.title('Evaluating Accuracy')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig('plot-acc.png')
        plt.show()

