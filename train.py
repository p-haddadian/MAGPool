import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from networks import Net
from utils import EarlyStopping


def arg_parse(args=None):
    parser = argparse.ArgumentParser(description='MAGPool')
    parser.add_argument('--dataset', type=str, default='FRANKENSTEIN',
                        help='DD/PROTEINS/NCI1/NCI109/FRANKENSTEIN')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of epochs')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='weight decay')
    parser.add_argument('--hidden', type=int, default=512, help='hidden size')
    parser.add_argument('--pooling_ratio', type=float,
                        default=0.35, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float,
                        default=0.2, help='dropout ratio')
    parser.add_argument('--num_heads', type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument('--alpha', type=float, default=.3, help='alpha')
    parser.add_argument("--hop_num", type=int, default=3, help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.0, help="p_norm")
    parser.add_argument('--early_stop', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=100,
                        help='patience for earlystopping')
    args = parser.parse_args(args)
    return args


def test(model, loader, args):
    '''
    Evaluation and test function
    Inputs:
        - model: PyTorch model
        - loader: dataloader corresponding to evaluation
        - args: specified arguments
    Outputs:
        - accuracy, loss
    '''
    model.eval()
    correct = 0
    loss = 0
    loss_fcn = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += loss_fcn(out, data.y).item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def main(args):
    args.device = 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:0'

    # loading the dataset
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    args.num_graphs = len(dataset)

    # data spliting
    num_training = int(len(dataset)*0.8)
    num_val = int(len(dataset)*0.1)
    num_test = len(dataset) - (num_training+num_val)
    training_set, validation_set, test_set = random_split(
        dataset, [num_training, num_val, num_test])

    # dataloader
    train_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # logger
    print("""----Data Statistics----
                 Dataset: %s
                 # of graphs: %d
                 # of classes: %d
                 # of features: %d
                 ---------------------
                 Train samples: %d
                 Validation samples: %d
                 Test samples: %d""" % (args.dataset, args.num_graphs, args.num_classes, args.num_features, len(training_set), len(validation_set), len(test_set)))

    model = Net(args).to(args.device)
    print("----Model Configuration and Parameters----\n", model)
    print(model.alpha)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop:
        stopper = EarlyStopping(args.patience)
    loss_fcn = torch.nn.CrossEntropyLoss()

    min_loss = 10e9
    patience = 0

    train_losses = list()
    val_losses = list()

    train_accs = list()
    val_accs = list()

    # Training the model
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        loss_all = 0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            loss = loss_fcn(out, data.y)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss_all += data.y.size(0) * loss.item()
            train_acc = correct / len(train_loader.dataset)
            print('Training Loss: {0:.4f} | Training Acc: {1:.4f}'.format(
                loss, train_acc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad
        val_acc, val_loss = test(model, val_loader, args)
        train_loss = loss_all / len(train_loader)
        print('Epoch: {0} | Train Loss: {1:.4f} | Val Loss: {2:.4f} | Val Acc: {3:.4f}'.format(
            epoch, train_loss, val_loss, val_acc))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print('Model saved at epoch {}'.format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            print(
                'Maximum patience reached at epoch {} and val loss had no change'.format(epoch))
            break

    # Testing the model
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader, args)
    print('---------------Test----------------')
    print('Test loss: {0:.4f} | Test Acc: {1:.4f}'.format(test_loss, test_acc))


    # Plotting the necessary metrics
    losses = [train_losses, val_losses]
    accs = [train_accs, val_accs]
    plotter(losses, accs)


if __name__ == '__main__':
    main(arg_parse())
