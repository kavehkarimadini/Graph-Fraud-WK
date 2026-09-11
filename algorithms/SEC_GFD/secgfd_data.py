"""Dataset loading for the SEC-GFD unified benchmark."""

import os

import dgl
import numpy as np
import torch
from dgl.data import FraudAmazonDataset, FraudYelpDataset

from utils.data_loader import load_tgraph_arrays


def canonical_dataset_name(name):
    name = str(name).lower().strip()

    aliases = {
        "yelpchi": "yelp",
        "yelp": "yelp",
        "amazon": "amazon",
        "tfinance": "tfinance",
        "t-finance": "tfinance",
        "tsocial": "tsocial",
        "t-social": "tsocial",
    }

    if name not in aliases:
        raise ValueError(
            "Unsupported SEC-GFD dataset '{}'. "
            "Use amazon, yelp/yelpchi, tfinance, or tsocial.".format(name)
        )

    return aliases[name]


def _flatten_labels(labels):
    labels = labels.long()

    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1)

    return labels.view(-1)


def _load_amazon_yelp(dataset_name, dataset_dir):
    if dataset_name == "amazon":
        dataset = FraudAmazonDataset(raw_dir=dataset_dir)
        labeled_from = 3305
    else:
        dataset = FraudYelpDataset(raw_dir=dataset_dir)
        labeled_from = None

    graph = dataset[0]

    # Match the released SEC-GFD implementation:
    # convert fraud heterograph to one homogeneous graph.
    graph = dgl.to_homogeneous(
        graph,
        ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
    )

    # Official SEC-GFD loader adds self-loops to Amazon/Yelp.
    graph = dgl.add_self_loop(graph)

    graph.ndata["feature"] = graph.ndata["feature"].float()
    graph.ndata["label"] = _flatten_labels(graph.ndata["label"])

    return graph, labeled_from


def _load_t_dataset(dataset_name, dataset_dir):
    features, labels, src, dst = load_tgraph_arrays(
        dataset_name,
        prefix=dataset_dir,
    )

    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    graph = dgl.graph(
        (
            torch.as_tensor(src, dtype=torch.int64),
            torch.as_tensor(dst, dtype=torch.int64),
        ),
        num_nodes=int(features.shape[0]),
    )

    graph.ndata["feature"] = torch.as_tensor(
        features,
        dtype=torch.float32,
    )

    graph.ndata["label"] = torch.as_tensor(
        labels,
        dtype=torch.long,
    ).view(-1)

    # Match released SEC-GFD code:
    # no self-loop is added to T-Finance/T-Social when del_ratio=0.
    return graph, None


def load_secgfd_dataset(dataset_name, project_root):
    """
    Load SEC-GFD data without PMP-specific feature normalization.

    Returns
    -------
    graph
        Homogeneous DGL graph.
    features
        Raw SEC-GFD input features as float32 tensor.
    labels
        Flat binary label tensor.
    eligible
        Nodes eligible for benchmark splitting.
    """

    dataset_name = canonical_dataset_name(dataset_name)
    dataset_dir = os.path.join(project_root, "dataset")

    if dataset_name in {"amazon", "yelp"}:
        graph, labeled_from = _load_amazon_yelp(
            dataset_name,
            dataset_dir,
        )

    elif dataset_name in {"tfinance", "tsocial"}:
        graph, labeled_from = _load_t_dataset(
            dataset_name,
            dataset_dir,
        )

    else:
        raise ValueError(
            "Unsupported SEC-GFD dataset: {}".format(dataset_name)
        )

    features = graph.ndata["feature"].float()
    labels = _flatten_labels(graph.ndata["label"])

    graph.ndata["feature"] = features
    graph.ndata["label"] = labels

    if labeled_from is None:
        eligible = np.arange(
            graph.num_nodes(),
            dtype=np.int64,
        )
    else:
        eligible = np.arange(
            int(labeled_from),
            graph.num_nodes(),
            dtype=np.int64,
        )

    y = labels.detach().cpu().numpy()

    eligible = eligible[
        np.isin(y[eligible], [0, 1])
    ]

    if eligible.size == 0:
        raise RuntimeError(
            "Dataset '{}' contains no eligible binary-labelled nodes."
            .format(dataset_name)
        )

    return graph, features, labels, eligible
