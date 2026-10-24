"""Dataset-specific CARE-GNN benchmark configuration."""

PAPER_SUPPORTED_DATASETS = {"yelp", "amazon"}


def canonical_dataset_name(name):
    name = str(name).lower().strip()

    aliases = {
        "yelp": "yelp",
        "yelpchi": "yelp",
        "amazon": "amazon",
        "tfinance": "tfinance",
        "t-finance": "tfinance",
        "elliptic": "elliptic",
        "fdcompcn": "fdcompcn",
        "comp": "fdcompcn",
    }

    if name not in aliases:
        raise ValueError(
            f"Unsupported CARE-GNN dataset: {name}"
        )

    return aliases[name]


CAREGNN_PAPER_CONFIG = {
    "yelp": {
        "hidden_dim": 64,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "lambda_1": 2.0,
        "step_size": 0.02,
        "initial_threshold": 0.5,
        "under_sampling_scale": 1.0,
        "num_layers": 1,
        "batch_size": 1024,
    },
    "amazon": {
        "hidden_dim": 64,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "lambda_1": 2.0,
        "step_size": 0.02,
        "initial_threshold": 0.5,
        "under_sampling_scale": 1.0,
        "num_layers": 1,
        "batch_size": 256,
    },
}


def get_caregnn_config(dataset):
    dataset = canonical_dataset_name(dataset)

    if dataset not in PAPER_SUPPORTED_DATASETS:
        raise ValueError(
            f"{dataset} has no CARE-GNN paper-defined configuration yet. "
            "Use validation-only paper-grounded tuning first."
        )

    config = dict(CAREGNN_PAPER_CONFIG[dataset])

    config["optimizer"] = "Adam"
    config["source_note"] = (
        "CARE-GNN original paper / official implementation settings."
    )

    return config