import numpy as np
import scipy.io as sio
import zipfile

from sklearn.model_selection import train_test_split

from utils.split import split_report, stratified_train_test_split
from utils.utils import pad_adjlist


def unzip_file(zip_src, dst_dir):
    iz = zipfile.is_zipfile(zip_src)
    if iz:
        zf = zipfile.ZipFile(zip_src, 'r')
        for file in zf.namelist():
            zf.extract(file, dst_dir)
    else:
        print('Zip Error.')


def _split_indices(y, test_size=0.4, seed=48):
    index = np.arange(len(y))
    train_idx, test_idx = stratified_train_test_split(y, test_size=test_size, seed=seed, indices=index)
    y = np.asarray(y)
    return train_idx, test_idx, y[train_idx], y[test_idx]


def load_data_dblp(path='../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat',
                   test_size=0.4, seed=48):
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]
    rownetworks = [data['net_APA'] - np.eye(N)]
    y = truelabels
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return rownetworks, features, X_train, y_train, X_test, y_test


def load_example_semi(test_size=0.2, seed=48):
    features = np.array([[1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1]])
    rownetworks = [np.array([[1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0]]),
                   np.array([[1, 0, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 1],
                             [1, 1, 0, 1, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1]])]
    y = np.array([[0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return rownetworks, features, X_train, y_train, X_test, y_test


def load_example_gem(test_size=0.2, seed=8):
    features = np.array([[5, 3, 0, 1, 0, 0, 0, 1, 0],
                         [2, 3, 1, 2, 0, 0, 0, 1, 0],
                         [3, 1, 6, 4, 0, 0, 1, 1, 0],
                         [0, 0, 2, 4, 4, 1, 0, 1, 1],
                         [0, 0, 3, 3, 1, 0, 1, 0, 1],
                         [1, 2, 5, 1, 4, 1, 0, 0, 1],
                         [0, 1, 3, 5, 1, 0, 0, 0, 1],
                         [0, 3, 4, 5, 2, 1, 1, 0, 1]])
    rownetworks = [np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])[:, np.newaxis]
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return rownetworks, features, X_train, y_train, X_test, y_test


def load_data_gas(test_size=0.4, seed=48):
    user_review_adj = pad_adjlist([[0, 1], [2], [3], [5], [4, 6]])
    user_item_adj = pad_adjlist([[0, 1], [0], [0], [2], [1, 2]])
    item_review_adj = pad_adjlist([[0, 2, 3], [1, 4], [5, 6]])
    item_user_adj = pad_adjlist([[0, 1, 2], [0, 4], [3, 4]])
    review_item_adj = [0, 1, 0, 0, 1, 2, 2]
    review_user_adj = [0, 0, 1, 2, 4, 3, 4]

    review_vecs = np.array([[1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1]])
    user_vecs = np.array([[1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1]])
    item_vecs = np.array([[1, 0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1]])
    features = [review_vecs, user_vecs, item_vecs]
    homo_adj = [[1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0]]
    adjs = [user_review_adj, user_item_adj, item_review_adj, item_user_adj,
            review_user_adj, review_item_adj, homo_adj]
    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return adjs, features, X_train, y_train, X_test, y_test


def get_split_info(all_labels, train_idx, test_idx, test_size, seed):
    info = split_report(all_labels, train_idx, test_idx)
    info['test_fraction'] = float(test_size)
    info['seed'] = int(seed)
    info['stratified'] = True
    return info
