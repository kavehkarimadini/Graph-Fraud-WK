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
        "fdcompcn": "fdcompcn",
    }

    if name not in aliases:
        raise ValueError(
            "Unsupported SEC-GFD dataset '{}'. "
            "Use amazon, yelp/yelpchi, tfinance, tsocial, elliptic, "
            "or fdcompcn.".format(name)
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
    "elliptic": {
        "hidden_dim": 16,
        "order": 2,
        "high_order": 2,
    },
}


def get_secgfd_config(dataset_name):
    dataset_name = canonical_dataset_name(
        dataset_name
    )

    if dataset_name == "fdcompcn":
        raise ValueError(
            "FDCompCN does not yet have a fixed SEC-GFD configuration. "
            "Run the predefined validation-only hyperparameter search "
            "first, select the configuration using mean validation AUROC "
            "across the five benchmark seeds, and freeze that configuration "
            "before final benchmark evaluation."
        )

    cfg = dict(
        SEC_GFD_CONFIG[
            dataset_name
        ]
    )

    if dataset_name == "elliptic":
        source_note = (
            "Elliptic SEC-GFD configuration selected by the predefined "
            "validation-only hyperparameter search. Search space: "
            "hidden_dim in {16, 64} and order_C in {2, 5}; "
            "all configurations evaluated on train_40 using the five "
            "predefined benchmark seeds and selected by mean validation "
            "AUROC. Selected hidden_dim=16 and order_C=2. "
            "The test set was not accessed during tuning. "
            "Learning rate=0.01, weight decay=0.0, lambda_env=0.2, "
            "Adam optimizer, and high_order=2 were kept fixed from "
            "SEC-GFD paper/released-code settings."
        )
    else:
        source_note = (
            "SEC-GFD paper/released repository settings; "
            "high_order=2 preserved from released implementation."
        )

    cfg.update(
        {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "lambda_env": 0.2,
            "optimizer": "Adam",
            "source_note": source_note,
        }
    )

    return cfg