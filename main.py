import argparse
import time
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '.')))

from algorithms.GEM.GEM import GEM
from algorithms.GeniePath.GeniePath import GeniePath
from algorithms.Player2Vec.Player2Vec import Player2Vec
from algorithms.FdGars.FdGars import FdGars
from algorithms.SemiGNN.SemiGNN import SemiGNN
from algorithms.GAS.GAS import GAS
from utils.data_loader import *
from utils.utils import *
from utils.torch_helpers import get_device, set_seed
from utils.training import add_save_args, finalize_run, resolve_test_size


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GAS',
                        help="['Player2Vec', 'FdGars','GEM','SemiGNN','GAS','GeniePath']")
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
    parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--hidden1', type=int, default=16)
    parser.add_argument('--hidden2', type=int, default=16)
    parser.add_argument('--gcn_output', type=int, default=4)
    parser.add_argument('--review_num sample', default=7)
    parser.add_argument('--gcn_dim', type=int, default=5)
    parser.add_argument('--encoding1', type=int, default=64)
    parser.add_argument('--encoding2', type=int, default=64)
    parser.add_argument('--encoding3', type=int, default=64)
    parser.add_argument('--encoding4', type=int, default=64)
    parser.add_argument('--init_emb_size', type=int, default=4)
    parser.add_argument('--semi_encoding1', type=int, default=3)
    parser.add_argument('--semi_encoding2', type=int, default=2)
    parser.add_argument('--semi_encoding3', type=int, default=4)
    parser.add_argument('--Ul', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lamtha', type=float, default=0.5)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--layer_num', type=int, default=4)
    add_save_args(parser)
    return parser.parse_args()


def get_batch(ix, int_batch, train_size, train_data, train_label):
    if ix + int_batch >= train_size:
        ix = max(train_size - int_batch, 0)
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def load_data(args):
    test_size = resolve_test_size(args, args.model)
    split_kw = dict(test_size=test_size, seed=args.seed)

    # GAS needs a User-Review-Item graph (load_data_gas), not DBLP's author graph.
    if args.model == 'GAS' and args.dataset_str != 'example':
        raise ValueError(
            "GAS only supports --dataset_str example (User-Review-Item toy graph). "
            "DBLP is an author-paper graph used by FdGars, Player2Vec, GeniePath, and GEM, "
            "and does not provide the review/user/item adjacencies GAS needs. "
            "Run: python main.py --model GAS --dataset_str example --epoch_num 100 "
            "--batch_size 4 --gpu 0"
        )

    if args.dataset_str == 'dblp':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp(
            'dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat', **split_kw)
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]
        split_info = get_split_info(
            _full_labels(train_data, train_label, test_data, test_label, node_size),
            train_data, test_data, test_size, args.seed)
    elif args.model == 'GAS':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_gas(**split_kw)
        node_embedding_r = features[0].shape[1]
        node_embedding_u = features[1].shape[1]
        node_embedding_i = features[2].shape[1]
        node_size = features[0].shape[0]
        h_u_size = adj_list[0].shape[1] * (node_embedding_r + node_embedding_u)
        h_i_size = adj_list[2].shape[1] * (node_embedding_r + node_embedding_i)
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding_r, node_embedding_u, node_embedding_i, class_size, train_size, h_u_size,
                 h_i_size]
        split_info = get_split_info(
            _gas_labels(node_size, train_data, train_label, test_data, test_label),
            train_data, test_data, test_size, args.seed)
    else:
        if args.model == 'GEM':
            adj_list, features, train_data, train_label, test_data, test_label = load_example_gem(**split_kw)
        else:
            adj_list, features, train_data, train_label, test_data, test_label = load_example_semi(**split_kw)
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]
        split_info = get_split_info(
            _full_labels(train_data, train_label, test_data, test_label, node_size),
            train_data, test_data, test_size, args.seed)

    return adj_list, features, train_data, train_label, test_data, test_label, paras, split_info


def _full_labels(train_data, train_label, test_data, test_label, n_nodes):
    labels = np.zeros((n_nodes, train_label.shape[1]))
    labels[train_data] = train_label
    labels[test_data] = test_label
    return labels


def _gas_labels(n_nodes, train_data, train_label, test_data, test_label):
    labels = np.zeros((n_nodes, train_label.shape[1]))
    labels[train_data] = train_label
    labels[test_data] = test_label
    return labels


def build_model(args, paras, adj_list, device):
    extra = {}
    if args.model == 'Player2Vec':
        adj_data = [normalize_adj(adj) for adj in adj_list]
        net = Player2Vec(class_size=paras[2], gcn_output1=args.hidden1, meta=len(adj_list),
                         nodes=paras[0], embedding=paras[1], encoding=args.gcn_output, device=device)
    elif args.model == 'FdGars':
        adj_data = [normalize_adj(adj) for adj in adj_list]
        net = FdGars(class_size=paras[2], gcn_output1=args.hidden1, gcn_output2=args.hidden2,
                     meta=len(adj_list), nodes=paras[0], embedding=paras[1], encoding=args.gcn_output,
                     device=device)
    elif args.model == 'GAS':
        adj_data = adj_list
        net = GAS(nodes=paras[0], class_size=paras[4], embedding_r=paras[1], embedding_u=paras[2],
                  embedding_i=paras[3], h_u_size=paras[6], h_i_size=paras[7],
                  encoding1=args.encoding1, encoding2=args.encoding2, encoding3=args.encoding3,
                  encoding4=args.encoding4, gcn_dim=args.gcn_dim, device=device)
    elif args.model == 'GEM':
        adj_data = adj_list
        net = GEM(class_size=paras[2], encoding=args.k, meta=len(adj_list), nodes=paras[0],
                  embedding=paras[1], hop=args.hop, device=device)
    elif args.model == 'GeniePath':
        adj_data = adj_list
        net = GeniePath(out_dim=paras[2], dim=args.dim, lstm_hidden=args.lstm_hidden,
                        nodes=paras[0], in_dim=paras[1], heads=args.heads, layer_num=args.layer_num,
                        class_size=paras[2], device=device)
    elif args.model == 'SemiGNN':
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
        extra['u_i'] = np.concatenate(np.array(u_i))
        extra['u_j'] = np.concatenate(np.array(u_j))
        extra['graph_label'] = np.concatenate(graph_labels, axis=0)
    else:
        raise ValueError('Unknown model: {}'.format(args.model))
    return net, adj_data, extra


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras, split_info):
    device = get_device(args.gpu)
    net, adj_data, extra = build_model(args, paras, adj_list, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    t_start = time.perf_counter()
    for epoch in range(args.epoch_num):
        train_loss = 0
        train_acc = 0
        count = 0
        for index in range(0, paras[3], args.batch_size):
            batch_data, batch_label = get_batch(index, args.batch_size, paras[3], train_data, train_label)
            if args.model == 'SemiGNN':
                loss, acc, pred, prob = net.fit_batch(
                    adj_data, extra['u_i'], extra['u_j'], extra['graph_label'],
                    batch_data, batch_label, optimizer=optimizer)
            else:
                loss, acc, pred, prob = net.fit_batch(
                    features, adj_data, batch_label, batch_data, optimizer=optimizer)
            print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
            train_loss += loss
            train_acc += acc
            count += 1
        print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(
            epoch, train_loss / count, train_acc / count))

    print("train time=", "{:.5f}".format(time.perf_counter() - t_start))
    print("Train end!")

    if args.model == 'SemiGNN':
        train_eval = lambda: net.evaluate(
            adj_data, extra['u_i'], extra['u_j'], extra['graph_label'], train_data, train_label)
        test_eval = lambda: net.evaluate(
            adj_data, extra['u_i'], extra['u_j'], extra['graph_label'], test_data, test_label)
    else:
        train_eval = lambda: net.evaluate(features, adj_data, train_label, train_data)
        test_eval = lambda: net.evaluate(features, adj_data, test_label, test_data)

    finalize_run(
        args, args.model, net, split_info,
        train_data, train_label, test_data, test_label,
        train_eval, test_eval, sigmoid=True,
    )


if __name__ == "__main__":
    args = arg_parser()
    set_seed(args.seed)
    adj_list, features, train_data, train_label, test_data, test_label, paras, split_info = load_data(args)
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras, split_info)
