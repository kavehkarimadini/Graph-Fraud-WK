'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.GAS.GAS import GAS
from utils.data_loader import load_data_gas
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
    parser.add_argument('--gcn_dim', type=int, default=5)
    parser.add_argument('--encoding1', type=int, default=64)
    parser.add_argument('--encoding2', type=int, default=64)
    parser.add_argument('--encoding3', type=int, default=64)
    parser.add_argument('--encoding4', type=int, default=64)
    add_save_args(parser)
    return parser.parse_args()


def _gas_labels(n_nodes, train_data, train_label, test_data, test_label):
    labels = np.zeros((n_nodes, train_label.shape[1]))
    labels[train_data] = train_label
    labels[test_data] = test_label
    return labels


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    device = get_device(args.gpu)
    net = GAS(nodes=paras[0], class_size=paras[4], embedding_r=paras[1], embedding_u=paras[2],
              embedding_i=paras[3], h_u_size=paras[6], h_i_size=paras[7],
              encoding1=args.encoding1, encoding2=args.encoding2, encoding3=args.encoding3,
              encoding4=args.encoding4, gcn_dim=args.gcn_dim, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    def fit_fn(batch_data, batch_label, opt):
        return net.fit_batch(features, adj_list, batch_label, batch_data, optimizer=opt)

    run_epochs(net, optimizer, args, paras[3], train_data, train_label, fit_fn)
    finish(
        net, args, 'GAS', None, adj_list,
        train_data, train_label, test_data, test_label, paras[0],
        lambda: net.evaluate(features, adj_list, train_label, train_data),
        lambda: net.evaluate(features, adj_list, test_label, test_data),
        sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    test_size = resolve_test_size(args, 'GAS')
    adj_list, features, train_data, train_label, test_data, test_label = load_data_gas(
        test_size=test_size, seed=args.seed)
    node_embedding_r = features[0].shape[1]
    node_embedding_u = features[1].shape[1]
    node_embedding_i = features[2].shape[1]
    node_size = features[0].shape[0]
    h_u_size = adj_list[0].shape[1] * (node_embedding_r + node_embedding_u)
    h_i_size = adj_list[2].shape[1] * (node_embedding_r + node_embedding_i)
    paras = [node_size, node_embedding_r, node_embedding_u, node_embedding_i,
             train_label.shape[1], len(train_data), h_u_size, h_i_size]
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
