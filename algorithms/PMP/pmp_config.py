"""Dataset-specific PMP model hyperparameters.

Benchmark-wide protocol settings (seeds, train ratios, max epochs, early stopping)
remain in PMP_benchmark.py so they are identical across datasets.
"""

PMP_CONFIGS = {
    # Preserve the already-completed Amazon benchmark exactly.
    "amazon": {
        "hidden_dim": 128,
        "dropout": 0.6,
        "num_layers": 1,
        "batch_size": 128,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "num_trans": 1,
        "aggregation": "mean",
        "relation_aggregation": "cat",
        "projection": True,
        "mlp_activation": "elu",
        "eval_batch_size": 4096,
        "weighted_loss": False,
        "source_note": "Amazon values frozen to completed Graph-Fraud-WK PMP benchmark.",
    },
    # Official PMP config/yelp.yml model/optimizer settings.
    "yelp": {
        "hidden_dim": 256,
        "dropout": 0.0,
        "num_layers": 1,
        "batch_size": 512,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "num_trans": 1,
        "aggregation": "mean",
        "relation_aggregation": "cat",
        "projection": True,
        "mlp_activation": "relu",
        "eval_batch_size": 4096,
        "weighted_loss": False,
        "source_note": "Official Xtra-Computing/PMP config/yelp.yml; benchmark-wide stopping protocol applied separately.",
    },
    # Official PMP config/tfinance.yml active LA-SAGE-S settings.
    "tfinance": {
    "hidden_dim": 64,
    "dropout": 0.4,
    "num_layers": 1,
    "batch_size": 256,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "num_trans": 1,
    "aggregation": "mean",
    "relation_aggregation": "cat",
    "projection": True,
    "mlp_activation": "relu",
    "eval_batch_size": 4096,
    "weighted_loss": False,
    "source_note": "PMP ICLR 2024 Appendix D.4 dataset-specific T-Finance setting; unified benchmark stopping protocol applied separately.",
   },
}


def canonical_dataset_name(name):
    name = name.lower().strip()
    aliases = {
        "yelpchi": "yelp",
        "t-finance": "tfinance",
        "t_finance": "tfinance",
    }
    return aliases.get(name, name)


def get_pmp_config(dataset):
    dataset = canonical_dataset_name(dataset)
    if dataset not in PMP_CONFIGS:
        supported = ", ".join(sorted(PMP_CONFIGS))
        raise ValueError(f"Unsupported PMP dataset '{dataset}'. Supported: {supported}")
    return dict(PMP_CONFIGS[dataset])
