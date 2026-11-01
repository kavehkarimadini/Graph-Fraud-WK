"""Dataset-specific FRAUDRE benchmark configuration.

Amazon and YelpChi:
    FRAUDRE paper / official implementation settings, except the unified
    benchmark invariants (Adam, learning_rate=0.01, max_epochs=500,
    patience=25).

T-Finance, Elliptic, FDCompCN:
    FRAUDRE has no dataset-specific settings for these datasets.
    Their final configurations must therefore be selected using the
    predefined validation-only paper-grounded search space below.
"""

PAPER_SUPPORTED_DATASETS = {
    "yelp",
    "amazon",
}

VALIDATION_TUNED_DATASETS = {
    "tfinance",
    "elliptic",
    "fdcompcn",
}


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
            "Unsupported FRAUDRE dataset: {}".format(name)
        )

    return aliases[name]


# ------------------------------------------------------------------
# Unified benchmark invariants
# ------------------------------------------------------------------

OPTIMIZER = "Adam"
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500
EARLY_STOP = 25

EMBED_DIM = 64
N_LAYERS = 2
HIDDEN_DIM = 64
MAX_NEIGHBORS = 10


# ------------------------------------------------------------------
# FRAUDRE paper-supported dataset configurations
# ------------------------------------------------------------------

FRAUDRE_PAPER_CONFIG = {
    "yelp": {
        "embed_dim": 64,
        "n_layers": 2,
        "hidden_dim": 64,
        "batch_size": 1024,
        "max_neighbors": 10,
        "weight_decay": 1e-4,
        "learning_rate": 0.01,
        "feature_normalisation": "row",
    },

    "amazon": {
        "embed_dim": 64,
        "n_layers": 2,
        "hidden_dim": 64,
        "batch_size": 256,
        "max_neighbors": 10,
        "weight_decay": 1e-4,
        "learning_rate": 0.01,
        "feature_normalisation": "row",
    },
}


# ------------------------------------------------------------------
# Predefined search for datasets without FRAUDRE-specific settings
# ------------------------------------------------------------------
#
# Paper-grounded values:
#   embedding size: FRAUDRE sensitivity analysis reports 16, 32, 64, 128
#   convolution layers: FRAUDRE sensitivity analysis compares 1 and 2
#   batch size: official experiments use 256 (Amazon) and 1024 (YelpChi)
#
# Selection protocol:
#   - train_40 only
#   - all five fixed benchmark seeds
#   - select highest mean validation AUROC
#   - test set must remain untouched
#
# The search space is defined BEFORE tuning.

FRAUDRE_TUNING_SPACE = {
    "embed_dim": [16, 32, 64, 128],
    "n_layers": [1, 2],
    "batch_size": [256, 1024],
}


def get_fraudre_config(dataset):
    dataset = canonical_dataset_name(dataset)

    if dataset not in PAPER_SUPPORTED_DATASETS:
        raise ValueError(
            "{} has no FRAUDRE paper-defined configuration yet. "
            "Run the predefined validation-only tuning procedure first."
            .format(dataset)
        )

    config = dict(
        FRAUDRE_PAPER_CONFIG[dataset]
    )

    config["optimizer"] = OPTIMIZER

    config["source_note"] = (
        "FRAUDRE paper / official implementation settings. "
        "Unified benchmark overrides the original dataset-specific "
        "learning rate with Adam learning_rate=0.01 and uses "
        "max_epochs=500 with validation-AUROC early stopping "
        "patience=25."
    )

    return config
