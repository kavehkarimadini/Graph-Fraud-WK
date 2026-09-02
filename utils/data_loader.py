import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


YELPCHI_RELATIONS = ('net_rur', 'net_rtr', 'net_rsr')
AMAZON_RELATIONS = ('net_upu', 'net_usu', 'net_uvu')


def _mat_user_keys(data):
    return [k for k in data.keys() if not k.startswith('__')]


def _load_mat_graph(path, relations, test_size=0.2, seed=48, dataset_name='dataset'):
    """Load a CARE-GNN-style .mat: features, label, and named adjacency matrices."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{0} file not found at {1}. Unzip dataset/{0}.zip first "
            "(Expand-Archive on Windows).".format(dataset_name, os.path.abspath(path)))
    data = sio.loadmat(path)
    truelabels = np.array(data['label']).reshape(-1).astype(int)
    features = data['features'].astype(float)
    if sp.issparse(features):
        features = np.asarray(features.todense())
    else:
        features = np.asarray(features)
    available = _mat_user_keys(data)
    adjs = []
    for relation in relations:
        if relation not in data:
            raise KeyError("Relation '{}' not in {}; available keys: {}".format(
                relation, os.path.basename(path), available))
        adj = data[relation]
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        adjs.append(adj)
    n_classes = int(truelabels.max()) + 1
    y = np.eye(n_classes, dtype=float)[truelabels]
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return adjs, features, X_train, y_train, X_test, y_test


def load_data_yelpchi(path='../../dataset/YelpChi.mat', test_size=0.2, seed=48,
                      relation='net_rur'):
    """Homogeneous YelpChi graph (review-user-review) for GCN models such as FdGars."""
    return _load_mat_graph(path, [relation], test_size=test_size, seed=seed,
                           dataset_name='YelpChi')


def load_data_yelpchi_multi(path='../../dataset/YelpChi.mat', test_size=0.2, seed=48,
                            relations=None):
    """Multi-relation YelpChi (R-U-R, R-T-R, R-S-R)."""
    if relations is None:
        relations = list(YELPCHI_RELATIONS)
    return _load_mat_graph(path, relations, test_size=test_size, seed=seed,
                           dataset_name='YelpChi')


def load_data_amazon(path='../../dataset/Amazon.mat', test_size=0.2, seed=48,
                     relation='homo'):
    """Homogeneous Amazon graph (union of relations) for GCN models such as FdGars."""
    return _load_mat_graph(path, [relation], test_size=test_size, seed=seed,
                           dataset_name='Amazon')


def load_data_amazon_multi(path='../../dataset/Amazon.mat', test_size=0.2, seed=48,
                           relations=None):
    """Multi-relation Amazon (U-P-U, U-S-U, U-V-U) for GraphConsis-style models."""
    if relations is None:
        relations = list(AMAZON_RELATIONS)
    return _load_mat_graph(path, relations, test_size=test_size, seed=seed,
                           dataset_name='Amazon')


def resolve_tgraph_path(name, prefix=None):
    """Locate a DGL binary such as dataset/tfinance or dataset/tsocial (no extension)."""
    name = os.path.basename(str(name)).replace('.zip', '').replace('.bin', '')
    candidates = []
    if prefix:
        candidates.append(os.path.join(prefix, name))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join('dataset', name),
        os.path.join(here, '..', 'dataset', name),
        os.path.join(os.getcwd(), 'dataset', name),
        os.path.join(os.getcwd(), '..', '..', 'dataset', name),
        name,
    ])
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "DGL graph '{}' not found. Unzip dataset/{}.zip into dataset/ "
        "(the extracted file has no extension).".format(name, name))


def load_tgraph_arrays(name, prefix=None):
    """Load T-Finance / T-Social without DGL: features, labels, src, dst (numpy)."""
    from utils.dgl_bin import load_dgl_v2_homo_arrays
    path = resolve_tgraph_path(name, prefix=prefix)
    return load_dgl_v2_homo_arrays(path)


class HomoGraph(object):
    """Minimal DGL-like homogeneous graph (ndata, in_degrees, sparse adj)."""

    def __init__(self, src, dst, n, features, labels):
        import torch
        self.ndata = {
            'feature': torch.as_tensor(np.asarray(features), dtype=torch.float32),
            'label': torch.as_tensor(np.asarray(labels), dtype=torch.long).view(-1),
        }
        self._src = torch.as_tensor(np.asarray(src), dtype=torch.long)
        self._dst = torch.as_tensor(np.asarray(dst), dtype=torch.long)
        self._n = int(n)
        self._adj = None
        self._in_deg = torch.bincount(self._dst, minlength=self._n)

    def num_nodes(self):
        return self._n

    def in_degrees(self):
        return self._in_deg

    def edges(self):
        return self._src, self._dst

    @property
    def adj(self):
        import torch
        if self._adj is None:
            idx = torch.stack([self._dst, self._src], dim=0)
            val = torch.ones(self._src.numel(), dtype=torch.float32, device=self._src.device)
            self._adj = torch.sparse_coo_tensor(
                idx, val, (self._n, self._n), device=self._src.device).coalesce()
        return self._adj

    def to(self, device):
        self.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self._in_deg = self._in_deg.to(device)
        self._adj = None
        return self


def load_tgraph_dgl(name, prefix=None):
    """Load T-Finance / T-Social as a HomoGraph (DGL v2 binary, no DGL import)."""
    features, labels, src, dst = load_tgraph_arrays(name, prefix=prefix)
    return HomoGraph(src, dst, features.shape[0], features, labels)


def load_data_tfinance(path='../../dataset/tfinance', test_size=0.4, seed=48):
    """Homogeneous T-Finance graph for GCN-style models (one-hot labels)."""
    return _load_tgraph_numpy(path, test_size=test_size, seed=seed)


def load_data_tsocial(path='../../dataset/tsocial', test_size=0.4, seed=48):
    """Homogeneous T-Social graph for GCN-style models (one-hot labels)."""
    return _load_tgraph_numpy(path, test_size=test_size, seed=seed)


def _load_tgraph_numpy(path, test_size=0.4, seed=48):
    features, truelabels, src, dst = load_tgraph_arrays(path)
    n = features.shape[0]
    adj = sp.coo_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    n_classes = int(truelabels.max()) + 1
    y = np.eye(n_classes, dtype=float)[truelabels]
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return [adj], features, X_train, y_train, X_test, y_test


def load_tgraph_pyg(name, prefix=None, test_size=0.2, seed=48, normalize=True):
    """T-Finance / T-Social as PyG tensors for GraphConsis (single relation)."""
    import torch

    feat, labels, src, dst = load_tgraph_arrays(name, prefix=prefix)
    edge_index = torch.stack(
        [torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)], dim=0)
    if normalize:
        feat = StandardScaler().fit_transform(feat)
    x = torch.tensor(np.asarray(feat), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_idx, test_idx = stratified_train_test_split(labels, test_size=test_size, seed=seed)
    train_idx_t = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    test_idx_t = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    split_info = split_report(labels, train_idx, test_idx)
    split_info.update({'test_fraction': float(test_size), 'seed': int(seed), 'stratified': True})
    return x, [edge_index], y, train_idx_t, test_idx_t, split_info


def scipy_to_dgl(adj, features, labels):
    """Build a homogeneous graph from a scipy adjacency matrix (BWGNN)."""
    coo = sp.coo_matrix(adj) if not sp.issparse(adj) else adj.tocoo()
    y = np.asarray(labels)
    if y.ndim == 2 and y.shape[1] > 1:
        y = y.argmax(axis=1)
    return HomoGraph(coo.row, coo.col, np.asarray(features).shape[0], features, y.reshape(-1))


ELLIPTIC_CSV_NAMES = (
    'elliptic_txs_classes.csv',
    'elliptic_txs_edgelist.csv',
    'elliptic_txs_features.csv',
)


def is_elliptic_name(name):
    key = os.path.basename(str(name)).lower().replace('.zip', '').replace('.mat', '')
    return key in ('elliptic', 'elliptic_bitcoin', 'elliptic_bitcoin_dataset')


def resolve_elliptic_dir(prefix=None):
    """Locate ``dataset/elliptic_bitcoin_dataset`` (the folder with the three CSVs)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if prefix:
        prefix = os.path.normpath(prefix)
        candidates.extend([
            prefix,
            os.path.join(prefix, 'elliptic_bitcoin_dataset'),
            os.path.join(prefix, 'elliptic'),
        ])
    candidates.extend([
        os.path.join('dataset', 'elliptic_bitcoin_dataset'),
        os.path.join(here, '..', 'dataset', 'elliptic_bitcoin_dataset'),
        os.path.join(os.getcwd(), 'dataset', 'elliptic_bitcoin_dataset'),
        os.path.join(os.getcwd(), '..', '..', 'dataset', 'elliptic_bitcoin_dataset'),
    ])
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isdir(path) and all(os.path.isfile(os.path.join(path, n)) for n in ELLIPTIC_CSV_NAMES):
            return path
    raise FileNotFoundError(
        "Elliptic Bitcoin CSVs not found. Expected folder dataset/elliptic_bitcoin_dataset "
        "with elliptic_txs_classes.csv, elliptic_txs_edgelist.csv, elliptic_txs_features.csv.")


def load_elliptic_arrays(path=None, prefix=None, labeled_only=True):
    """Load Elliptic Bitcoin (Weber et al., 2019) as numpy ``(features, labels, src, dst)``.

    By default only **labeled** transactions are kept (class ``1`` illicit, ``2`` licit;
    ``unknown`` is dropped) and the payment graph is induced on those nodes. Features
    are timestep plus the 165 published local/aggregated descriptors (166-d).
    Labels are binary: ``1`` illicit, ``0`` licit.
    """
    import csv

    ddir = path if path and os.path.isdir(path) else resolve_elliptic_dir(prefix)
    class_path = os.path.join(ddir, 'elliptic_txs_classes.csv')
    edge_path = os.path.join(ddir, 'elliptic_txs_edgelist.csv')
    feat_path = os.path.join(ddir, 'elliptic_txs_features.csv')

    raw_label = {}
    with open(class_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label[int(row['txId'])] = str(row['class']).strip().lower()

    feat = np.loadtxt(feat_path, delimiter=',')
    feat_ids = feat[:, 0].astype(np.int64)
    feat_x = feat[:, 1:].astype(np.float32, copy=False)
    id_to_feat = {int(tid): i for i, tid in enumerate(feat_ids)}

    if labeled_only:
        keep = [tid for tid, c in raw_label.items() if c in ('1', '2') and tid in id_to_feat]
    else:
        keep = [tid for tid in feat_ids.tolist() if tid in raw_label]
    keep = np.asarray(keep, dtype=np.int64)
    n = int(keep.size)
    feat_rows = np.fromiter((id_to_feat[int(tid)] for tid in keep), dtype=np.int64, count=n)
    features = feat_x[feat_rows]
    labels = np.empty(n, dtype=np.int64)
    for i, tid in enumerate(keep):
        labels[i] = 1 if raw_label[int(tid)] == '1' else 0
    old_to_new = {int(tid): i for i, tid in enumerate(keep)}

    src_list, dst_list = [], []
    with open(edge_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            a, b = int(row['txId1']), int(row['txId2'])
            if a in old_to_new and b in old_to_new:
                src_list.append(old_to_new[a])
                dst_list.append(old_to_new[b])
    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    n_illicit = int((labels == 1).sum())
    print('Elliptic labeled-only={}, nodes={}, illicit={}, licit={}, edges={}, feat_dim={}'.format(
        labeled_only, n, n_illicit, n - n_illicit, int(src.size), features.shape[1]))
    return features, labels, src, dst


def load_data_elliptic(path=None, prefix=None, test_size=0.4, seed=48, labeled_only=True):
    """Homogeneous Elliptic graph for GCN-style models (one-hot labels)."""
    features, truelabels, src, dst = load_elliptic_arrays(path=path, prefix=prefix, labeled_only=labeled_only)
    n = features.shape[0]
    adj = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    n_classes = int(truelabels.max()) + 1
    y = np.eye(n_classes, dtype=float)[truelabels]
    X_train, X_test, y_train, y_test = _split_indices(y, test_size=test_size, seed=seed)
    return [adj], features, X_train, y_train, X_test, y_test


def load_elliptic_pyg(prefix=None, test_size=0.4, seed=48, normalize=True, labeled_only=True):
    """Elliptic Bitcoin as PyG tensors for GraphConsis (single relation)."""
    import torch

    feat, labels, src, dst = load_elliptic_arrays(prefix=prefix, labeled_only=labeled_only)
    edge_index = torch.stack(
        [torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)], dim=0)
    if normalize:
        feat = StandardScaler().fit_transform(feat)
    x = torch.tensor(np.asarray(feat), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_idx, test_idx = stratified_train_test_split(labels, test_size=test_size, seed=seed)
    train_idx_t = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    test_idx_t = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    split_info = split_report(labels, train_idx, test_idx)
    split_info.update({'test_fraction': float(test_size), 'seed': int(seed), 'stratified': True})
    return x, [edge_index], y, train_idx_t, test_idx_t, split_info


def load_elliptic_graph(prefix=None, labeled_only=True):
    """Elliptic Bitcoin as a HomoGraph (BWGNN)."""
    features, labels, src, dst = load_elliptic_arrays(prefix=prefix, labeled_only=labeled_only)
    return HomoGraph(src, dst, features.shape[0], features, labels)


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
