"""Dataset-specific CARE-GNN benchmark configuration."""

PAPER_SUPPORTED_DATASETS = {"yelp", "amazon", "tfinance"}


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
    "tfinance": {
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
}


def get_caregnn_config(dataset):
    dataset = canonical_dataset_name(dataset)

    if dataset not in PAPER_SUPPORTED_DATASETS:
        raise ValueError(
            f"{dataset} has no CARE-GNN paper-defined or "
            "validation-selected configuration yet."
        )

    config = dict(CAREGNN_PAPER_CONFIG[dataset])
    config["optimizer"] = "Adam"

    if dataset == "tfinance":
        config["source_note"] = (
            "CARE-GNN T-Finance configuration selected by the predefined "
            "validation-only hyperparameter search. Batch sizes {256, 1024} "
            "were evaluated on train_40 using the five fixed benchmark seeds "
            "and selected by mean validation AUROC. Selected batch_size=1024. "
            "The test set was not accessed during tuning. Other hyperparameters "
            "were fixed to CARE-GNN paper / official implementation settings."
        )
    else:
        config["source_note"] = (
            "CARE-GNN original paper / official implementation settings."
        )

    return config
