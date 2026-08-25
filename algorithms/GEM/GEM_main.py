'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.GEM.GEM import GEM
from utils.data_loader import load_data_dblp, load_example_gem
from utils.standalone_train import finish, run_epochs
from utils.training import add_save_args, resolve_test_size
from utils.torch_helpers import get_device, set_seed


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_str', type=str, default='example')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--k', type=int, default=16)
    add_save_args(parser)
    return parser.parse_args()


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    device = get_device(args.gpu)
    net = GEM(class_size=paras[2], encoding=args.k, meta=len(adj_list), nodes=paras[0],
              embedding=paras[1], hop=args.hop, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    def fit_fn(batch_data, batch_label, opt):
        return net.fit_batch(features, adj_list, batch_label, batch_data, optimizer=opt)

    run_epochs(net, optimizer, args, paras[3], train_data, train_label, fit_fn)
    finish(
        net, args, 'GEM', None, adj_list,
        train_data, train_label, test_data, test_label, paras[0],
        lambda: net.evaluate(features, adj_list, train_label, train_data),
        lambda: net.evaluate(features, adj_list, test_label, test_data),
        sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    test_size = resolve_test_size(args, 'GEM')
    split_kw = dict(test_size=test_size, seed=args.seed)
    if args.dataset_str == 'dblp':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp(**split_kw)
    else:
        adj_list, features, train_data, train_label, test_data, test_label = load_example_gem(**split_kw)
    paras = [features.shape[0], features.shape[1], train_label.shape[1], len(train_data)]
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
