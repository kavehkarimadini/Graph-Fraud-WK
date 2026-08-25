'''Shared standalone training script helpers for algorithm *_main.py files.'''
import time

from utils.data_loader import get_split_info
from utils.training import finalize_run, resolve_test_size


def get_batch(ix, int_batch, train_size, train_data, train_label):
    if ix + int_batch >= train_size:
        ix = max(train_size - int_batch, 0)
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def run_epochs(net, optimizer, args, paras_train_size, train_data, train_label, fit_fn):
    t_start = time.perf_counter()
    for epoch in range(args.epoch_num):
        train_loss = train_acc = count = 0
        for index in range(0, paras_train_size, args.batch_size):
            batch_data, batch_label = get_batch(index, args.batch_size, paras_train_size, train_data, train_label)
            loss, acc, pred, prob = fit_fn(batch_data, batch_label, optimizer)
            print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
            train_loss += loss
            train_acc += acc
            count += 1
        print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(
            epoch, train_loss / max(count, 1), train_acc / max(count, 1)))
    print("train time=", "{:.5f}".format(time.perf_counter() - t_start))
    print("Train end!")


def full_labels_from_split(n_nodes, train_data, train_label, test_data, test_label):
    labels = __import__('numpy').zeros((n_nodes,) + tuple(train_label.shape[1:]))
    labels[train_data] = train_label
    labels[test_data] = test_label
    return labels


def finish(net, args, model_name, _features_or_none, _adj_data,
           train_data, train_label, test_data, test_label, n_nodes,
           train_eval_fn, test_eval_fn, sigmoid=True, extra_config=None):
    split_info = get_split_info(
        full_labels_from_split(n_nodes, train_data, train_label, test_data, test_label),
        train_data, test_data,
        resolve_test_size(args, model_name), args.seed,
    )
    return finalize_run(
        args, model_name, net, split_info,
        train_data, train_label, test_data, test_label,
        train_eval_fn, test_eval_fn, sigmoid=sigmoid, extra_config=extra_config,
    )
