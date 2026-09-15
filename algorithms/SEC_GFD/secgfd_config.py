"""Dataset-specific SEC-GFD benchmark configuration."""


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
        "elliptic": "elliptic",
    }

    if name not in aliases:
        raise ValueError(
            "Unsupported SEC-GFD dataset '{}'. "
            "Use amazon, yelp/yelpchi, tfinance, tsocial, or elliptic."
            .format(name)
        )

    return aliases[name]


SEC_GFD_CONFIG = {
    "amazon": {
        "hidden_dim": 64,
        "order": 2,
        "high_order": 2,
    },
    "yelp": {
        "hidden_dim": 64,
        "order": 2,
        "high_order": 2,
    },
    "tfinance": {
        "hidden_dim": 64,
        "order": 2,
        "high_order": 2,
    },
    "tsocial": {
        "hidden_dim": 16,
        "order": 5,
        "high_order": 2,
    },
}


def get_secgfd_config(dataset_name):
    dataset_name = canonical_dataset_name(dataset_name)

    if dataset_name == "elliptic":
        raise ValueError(
            "Elliptic does not yet have a fixed SEC-GFD configuration. "
            "Run the validation-only hyperparameter search first, then "
            "freeze the selected configuration before final benchmarking."
        )

    cfg = dict(SEC_GFD_CONFIG[dataset_name])

    cfg.update({
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "lambda_env": 0.2,
        "optimizer": "Adam",
        "source_note": (
            "SEC-GFD paper/released repository settings; "
            "high_order=2 preserved from released implementation."
        ),
    })

    return cfg