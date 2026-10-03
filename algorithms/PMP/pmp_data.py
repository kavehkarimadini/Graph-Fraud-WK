"""Shared dataset interface for PMP benchmark runs."""
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
    row_sum = torch.where(
        torch.isfinite(row_sum),
        row_sum,
        torch.ones_like(row_sum),
    )
    return features / row_sum


def _flatten_labels(labels):
    """Convert labels to a flat integer class vector."""
    labels = labels.long()

    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1)

    return labels.squeeze()


def _build_dgl_heterograph(features, labels, relations):
    """
    Build a single-node-type DGL heterograph from repository arrays.

    relations:
        list of (relation_name, src, dst)
    """
    num_nodes = int(features.shape[0])

    graph_data = {}

    for relation_name, src, dst in relations:
        src = torch.as_tensor(
            np.asarray(src, dtype=np.int64),
            dtype=torch.int64,
        )
        dst = torch.as_tensor(
            np.asarray(dst, dtype=np.int64),
            dtype=torch.int64,
        )

        graph_data[
            ("node", str(relation_name), "node")
        ] = (src, dst)

    if not graph_data:
        raise RuntimeError("Cannot construct PMP graph: no relations found.")

    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"node": num_nodes},
    )

    graph.ndata["feature"] = torch.as_tensor(
        np.asarray(features),
        dtype=torch.float32,
    )

    graph.ndata["label"] = torch.as_tensor(
        np.asarray(labels),
        dtype=torch.long,
    ).view(-1)

    return graph


def _load_repo_dataset(dataset_name, dataset_dir):
    """
    Load datasets already supported by Graph-Fraud-WK's shared loader.

    Used for:
        T-Finance
        T-Social
        Elliptic
        FDCompCN
    """
    from utils.data_loader import load_dga_arrays

    loader_name = {
        "tfinance": "tfinance",
        "tsocial": "tsocial",
        "elliptic": "elliptic",
        "fdcompcn": "comp",
    }[dataset_name]

    features, labels, relations, labeled_from = load_dga_arrays(
        name=loader_name,
        prefix=dataset_dir,
    )

    graph = _build_dgl_heterograph(
        features=features,
        labels=labels,
        relations=relations,
    )

    return graph, labeled_from


def load_pmp_dataset(dataset_name, project_root):
    """
    Load a dataset for PMP.

    Returns
    -------
    graph
        DGL heterograph.
    features
        Row-normalized float tensor.
    labels
        Flat binary label tensor.
    relations
        DGL edge relation names.
    eligible
        Nodes eligible for benchmark splitting.

    Supported datasets
    ------------------
    amazon
    yelp
    tfinance
    tsocial
    elliptic
    fdcompcn
    """

    dataset_name = canonical_dataset_name(dataset_name)
    dataset_dir = os.path.join(project_root, "dataset")

    # Amazon / Yelp:
    # preserve the exact pathway used for the completed Amazon benchmark
    # and the completed Yelp benchmark.
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

        # Amazon's first 3305 nodes are the unlabeled portion used
        # by the standard fraud dataset.
        labeled_from = 3305 if dataset_name == "amazon" else None

    # Repository-supported datasets
    elif dataset_name in {
        "tfinance",
        "tsocial",
        "elliptic",
        "fdcompcn",
    }:
        graph, labeled_from = _load_repo_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
        )

    else:
        raise ValueError(
            "Unsupported PMP dataset '{}'. "
            "Supported datasets: amazon, yelp, tfinance, tsocial, "
            "elliptic, fdcompcn.".format(dataset_name)
        )

    # Validate node data
    if "feature" not in graph.ndata:
        raise RuntimeError(
            "{} graph does not contain ndata['feature'].".format(
                dataset_name
            )
        )

    if "label" not in graph.ndata:
        raise RuntimeError(
            "{} graph does not contain ndata['label'].".format(
                dataset_name
            )
        )

    # Normalize features and labels
    features = row_normalize_features(
        graph.ndata["feature"]
    )

    labels = _flatten_labels(
        graph.ndata["label"]
    )

    graph.ndata["feature"] = features
    graph.ndata["label"] = labels

    # Relations
    relations = list(graph.etypes)

    if not relations:
        raise RuntimeError(
            "No edge relations found for dataset '{}'.".format(
                dataset_name
            )
        )

    # Eligible benchmark nodes
    if labeled_from is not None:
        eligible = np.arange(
            int(labeled_from),
            graph.num_nodes(),
            dtype=np.int64,
        )
    else:
        eligible = np.arange(
            graph.num_nodes(),
            dtype=np.int64,
        )

    # Only genuinely binary-labelled nodes enter benchmark splits.
    y = labels.detach().cpu().numpy()

    eligible = eligible[
        np.isin(y[eligible], [0, 1])
    ]

    if eligible.size == 0:
        raise RuntimeError(
            "Dataset '{}' contains no eligible binary-labelled nodes."
            .format(dataset_name)
        )

    return (
        graph,
        features,
        labels,
        relations,
        eligible,
    )