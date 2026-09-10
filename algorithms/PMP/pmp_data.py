"""Shared dataset interface for PMP benchmark runs."""
import os

import dgl
import numpy as np
import torch

from algorithms.PMP.pmp_config import canonical_dataset_name


def row_normalize_features(features):
    """Reproduce PMP feature row-normalization: x / (sum(x) + 0.01)."""
    features = features.float()
    row_sum = features.sum(dim=1, keepdim=True) + 0.01
    row_sum = torch.where(torch.isfinite(row_sum), row_sum, torch.ones_like(row_sum))
    return features / row_sum


def _flatten_labels(labels):
    labels = labels.long()
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1)
    return labels.squeeze()


def load_pmp_dataset(dataset_name, project_root):
    """Return graph, normalized features, labels, relation names, eligible nodes."""
    dataset_name = canonical_dataset_name(dataset_name)
    dataset_dir = os.path.join(project_root, "dataset")

    if dataset_name in {"amazon", "yelp"}:
        from dgl.data import FraudDataset

        dataset = FraudDataset(
            name=dataset_name,
            raw_dir=dataset_dir,
            train_size=0.4,
            val_size=0.2,
            random_seed=717,
            force_reload=False,
        )
        graph = dataset[0]

    elif dataset_name == "tfinance":
        candidates = [
            os.path.join(dataset_dir, "tfinance"),
            os.path.join(dataset_dir, "tfinance.bin"),
        ]
        graph_path = next((p for p in candidates if os.path.exists(p)), None)
        if graph_path is None:
            raise FileNotFoundError(
                "T-Finance graph not found. Expected dataset/tfinance (official DGL binary)."
            )
        graphs, _ = dgl.load_graphs(graph_path)
        graph = graphs[0]
    else:
        raise ValueError(f"Unsupported PMP dataset: {dataset_name}")

    if "feature" not in graph.ndata or "label" not in graph.ndata:
        raise RuntimeError(
            f"{dataset_name} graph must contain ndata['feature'] and ndata['label']."
        )

    features = row_normalize_features(graph.ndata["feature"])
    labels = _flatten_labels(graph.ndata["label"])
    graph.ndata["feature"] = features
    graph.ndata["label"] = labels

    relations = list(graph.etypes)
    if not relations:
        raise RuntimeError(f"No edge relation found in {dataset_name} graph.")

    # Amazon's first 3305 nodes are the unlabeled portion used by the DGL fraud dataset.
    # Yelp and T-Finance have labels for the full graph used in this benchmark.
    if dataset_name == "amazon":
        eligible = np.arange(3305, graph.num_nodes(), dtype=np.int64)
    else:
        eligible = np.arange(graph.num_nodes(), dtype=np.int64)

    y = labels.cpu().numpy()
    eligible = eligible[np.isin(y[eligible], [0, 1])]

    return graph, features, labels, relations, eligible
