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
from algorithms.FdGars.FdGars import FdGars
from utils.data_loader import load_data_dblp, load_data_yelpchi
from utils.standalone_train import finish, run_epochs
from utils.training import add_save_args, resolve_test_size
from utils.utils import normalize_adj
from utils.torch_helpers import get_device, set_seed


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_str', type=str, default='dblp',
                        help="['dblp','yelpchi']")
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--hidden1', type=int, default=16)
    parser.add_argument('--hidden2', type=int, default=16)
    parser.add_argument('--gcn_output', type=int, default=4)
    add_save_args(parser)
    return parser.parse_args()


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    device = get_device(args.gpu)
    adj_data = [normalize_adj(adj) for adj in adj_list]
    net = FdGars(class_size=paras[2], gcn_output1=args.hidden1, gcn_output2=args.hidden2,
                 meta=len(adj_list), nodes=paras[0], embedding=paras[1], encoding=args.gcn_output,
                 device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    def fit_fn(batch_data, batch_label, opt):
        return net.fit_batch(features, adj_data, batch_label, batch_data, optimizer=opt)

    run_epochs(net, optimizer, args, paras[3], train_data, train_label, fit_fn)
    finish(
        net, args, 'FdGars', None, adj_data,
        train_data, train_label, test_data, test_label, paras[0],
        lambda: net.evaluate(features, adj_data, train_label, train_data),
        lambda: net.evaluate(features, adj_data, test_label, test_data),
        sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    test_size = resolve_test_size(args, 'FdGars')
    split_kw = dict(test_size=test_size, seed=args.seed)
    ds = args.dataset_str.lower()
    if ds in ('yelp', 'yelpchi'):
        adj_list, features, train_data, train_label, test_data, test_label = load_data_yelpchi(
            '../../dataset/YelpChi.mat', **split_kw)
    elif ds == 'dblp':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp(
            **split_kw)
    else:
        raise ValueError(
            "FdGars supports --dataset_str dblp or yelpchi, not '{}'.".format(args.dataset_str))
    paras = [features.shape[0], features.shape[1], train_label.shape[1], len(train_data)]
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
