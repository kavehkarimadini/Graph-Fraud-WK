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
from algorithms.SemiGNN.SemiGNN import SemiGNN
from utils.data_loader import load_example_semi
from utils.standalone_train import finish, run_epochs
from utils.training import add_save_args, resolve_test_size
from utils.utils import get_negative_sampling, matrix_to_adjlist, pairs_to_matrix, random_walks
from utils.torch_helpers import get_device, set_seed


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_str', type=str, default='example')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--init_emb_size', type=int, default=4)
    parser.add_argument('--semi_encoding1', type=int, default=3)
    parser.add_argument('--semi_encoding2', type=int, default=2)
    parser.add_argument('--semi_encoding3', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lamtha', type=float, default=0.5)
    add_save_args(parser)
    return parser.parse_args()


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    device = get_device(args.gpu)
    adj_nodelists = [matrix_to_adjlist(adj, pad=False) for adj in adj_list]
    pairs = [random_walks(adj_nodelists[i], 2, 3) for i in range(len(adj_list))]
    net = SemiGNN(class_size=paras[2], semi_encoding1=args.semi_encoding1,
                  semi_encoding2=args.semi_encoding2, semi_encoding3=args.semi_encoding3,
                  meta=len(adj_list), nodes=paras[0], init_emb_size=args.init_emb_size,
                  ul=args.batch_size, alpha=args.alpha, lamtha=args.lamtha, device=device)
    adj_data = [pairs_to_matrix(p, paras[0]) for p in pairs]
    u_i, u_j, graph_labels = [], [], []
    for adj_nodelist, p in zip(adj_nodelists, pairs):
        u_i_t, u_j_t, graph_label = get_negative_sampling(p, adj_nodelist)
        u_i.append(u_i_t)
        u_j.append(u_j_t)
        graph_labels.append(graph_label)
    u_i = np.concatenate(np.array(u_i))
    u_j = np.concatenate(np.array(u_j))
    graph_label = np.concatenate(graph_labels, axis=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    def fit_fn(batch_data, batch_sup_label, opt):
        return net.fit_batch(adj_data, u_i, u_j, graph_label, batch_data, batch_sup_label, optimizer=opt)

    run_epochs(net, optimizer, args, paras[3], train_data, train_label, fit_fn)
    finish(
        net, args, 'SemiGNN', None, adj_data,
        train_data, train_label, test_data, test_label, paras[0],
        lambda: net.evaluate(adj_data, u_i, u_j, graph_label, train_data, train_label),
        lambda: net.evaluate(adj_data, u_i, u_j, graph_label, test_data, test_label),
        sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    test_size = resolve_test_size(args, 'SemiGNN')
    adj_list, features, train_data, train_label, test_data, test_label = load_example_semi(
        test_size=test_size, seed=args.seed)
    paras = [features.shape[0], features.shape[1], train_label.shape[1], len(train_data)]
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
