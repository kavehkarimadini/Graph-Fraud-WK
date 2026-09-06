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
from algorithms.GeniePath.GeniePath import GeniePath
from utils.data_loader import load_data_dblp
from utils.standalone_train import finish, run_epochs
from utils.training import add_save_args, resolve_test_size
from utils.torch_helpers import get_device, set_seed


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_str', type=str, default='dblp')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--layer_num', type=int, default=4)
    add_save_args(parser)
    return parser.parse_args()


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    device = get_device(args.gpu)
    net = GeniePath(out_dim=paras[2], dim=args.dim, lstm_hidden=args.lstm_hidden,
                    nodes=paras[0], in_dim=paras[1], heads=args.heads, layer_num=args.layer_num,
                    class_size=paras[2], device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    def fit_fn(batch_data, batch_label, opt):
        return net.fit_batch(features, adj_list, batch_label, batch_data, optimizer=opt)

    run_epochs(net, optimizer, args, paras[3], train_data, train_label, fit_fn)
    finish(
        net, args, 'GeniePath', None, adj_list,
        train_data, train_label, test_data, test_label, paras[0],
        lambda: net.evaluate(features, adj_list, train_label, train_data),
        lambda: net.evaluate(features, adj_list, test_label, test_data),
        sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    test_size = resolve_test_size(args, 'GeniePath')
    adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp(
        test_size=test_size, seed=args.seed)
    paras = [features.shape[0], features.shape[1], train_label.shape[1], len(train_data)]
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
