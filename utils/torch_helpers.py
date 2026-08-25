import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import dense_to_sparse, from_scipy_sparse_matrix


def get_device(gpu=None):
    if torch.cuda.is_available() and (gpu is None or str(gpu) != '-1'):
        if gpu is not None:
            return torch.device('cuda:{}'.format(gpu))
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        tensor = x.float()
    else:
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def to_long_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        tensor = x.long()
    else:
        tensor = torch.tensor(np.asarray(x), dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def adj_to_edge_index(adj, device=None):
    """Convert a dense or sparse adjacency matrix to PyG edge_index / edge_weight."""
    if sp.issparse(adj):
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_weight = edge_weight.float()
    else:
        adj_t = to_float_tensor(adj)
        edge_index, edge_weight = dense_to_sparse(adj_t)
    if device is not None:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    return edge_index, edge_weight


def adjs_to_edge_indices(adj_list, device=None):
    return [adj_to_edge_index(adj, device=device) for adj in adj_list]


def accuracy_from_logits(logits, labels):
    if labels.dim() > 1 and labels.size(-1) > 1:
        y_true = labels.argmax(dim=-1)
    else:
        y_true = labels.view(-1).long()
    y_pred = logits.argmax(dim=-1)
    return (y_pred == y_true).float().mean()


def pred_from_logits(logits, class_size, binary_threshold=0.5, binary=False):
    if binary:
        prob = torch.sigmoid(logits)
        return (prob > binary_threshold).float(), prob
    prob = torch.softmax(logits, dim=-1)
    pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=class_size).float()
    return pred, prob


def save_evaluation(save_dir, metrics, predictions=None, probabilities=None, model=None,
                    model_name='model'):
    """Persist evaluation metrics / predictions / checkpoint under save_dir."""
    import json
    import os

    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    if predictions is not None or probabilities is not None:
        np.savez_compressed(
            os.path.join(save_dir, 'predictions.npz'),
            predictions=np.asarray(predictions) if predictions is not None else np.array([]),
            probabilities=np.asarray(probabilities) if probabilities is not None else np.array([]),
        )

    if model is not None and hasattr(model, 'save'):
        model.save(os.path.join(save_dir, '{}.pt'.format(model_name)))
    elif model is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, '{}.pt'.format(model_name)))

    print('Evaluation saved to: {}'.format(os.path.abspath(save_dir)))
    print('  - metrics.json')
    if predictions is not None or probabilities is not None:
        print('  - predictions.npz')
    if model is not None:
        print('  - {}.pt'.format(model_name))
    return save_dir
