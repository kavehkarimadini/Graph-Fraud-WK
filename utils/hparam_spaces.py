"""Random-search spaces for ``tune_hparams.py``.

Keys are the parameter names used by ``utils.notebook_runner._<model>_defaults``
(which are also the ``*_main.py`` argparse ``dest`` names). Every value list is
discrete so that searches are reproducible and the stored configs are readable.

Sources for the ranges:
  * shared ranges follow GADBench (NeurIPS 2023, App. C Table 8: lr ~ 10^U(-3,-1),
    dropout {0, .1, .2, .3}, hidden {16, 32, 64}), GLHAD Sec. 4.1.4 (weight decay
    {1e-4, 1e-3, 1e-2, 1e-1} for baselines without paper settings) and PMP App. D.4
    (lr {.01, .005, .001}, weight decay {0, 5e-5, 1e-4});
  * model-specific ranges follow each paper's own sensitivity / grid-search
    section where it has one (cited inline).

T-Social (5.8M nodes) uses narrower widths / batch sizes (as GLHAD does with
hidden {10} on T-Social) so that every trial fits in GPU memory.
"""

LR = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]   # GADBench 10^U(-3,-1), discretised
WD = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]                 # PMP {0, 5e-5, 1e-4} + GLHAD {1e-4..1e-2}
DROPOUT = [0.0, 0.1, 0.2, 0.3]                     # GADBench
HIDDEN = [16, 32, 64]                              # GADBench / GLHAD
BATCH = [256, 512, 1024]                           # PMP App. D.4 range (upper half)

SPACES = {
    'AMNet': dict(
        hidden=[16, 32, 64, 128], order=[2, 3, 4, 5, 7],        # Bernstein order M
        dropout=DROPOUT, beta=[0.5, 1.0],                       # marginal-loss weight
        learning_rate=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'BSL': dict(
        hid_dim=[32, 64, 96, 128], batch_size=[128, 256, 512],
        alpha=[0.2, 0.4, 0.6, 0.8, 1.0], beta=[0.2, 0.4, 0.6, 0.8, 1.0],  # BSL Fig. 4 grid
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'BWGNN': dict(
        hid_dim=HIDDEN, order=[2, 3, 4, 5],                     # wavelet order C
        learning_rate=LR, weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'CAREGNN': dict(
        emb_size=[16, 32, 64, 128],                             # CARE-GNN Sec. 4.5
        lambda_1=[0.5, 1.0, 2.0, 4.0], step_size=[0.01, 0.02, 0.05],
        batch_size=BATCH, learning_rate=[1e-3, 5e-3, 1e-2, 5e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]),
    'CGNN': dict(
        emb_size=[32, 64, 128], tau=[0.01, 0.1, 0.5, 0.8], lam=[0.5, 1.0, 2.0],
        beta=[0.5, 1.0, 2.0], batch_size=[16, 64, 256],
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'CIEGAD': dict(
        hidden=[8, 16, 32, 64], order=[2, 4, 6, 8, 10, 12],     # CIE-GAD Fig. 7 ranges
        gamma=[0.0, 0.5, 1.0, 1.5, 2.0], dropout=DROPOUT,
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[0.0, 5e-5, 1e-4, 1e-3]),
    'COFRAUD': dict(
        hidden=[16, 32, 64, 128], n_layers=[1, 2],              # COFRAUD Table 3
        n_heads=[2, 4, 8], dropout=[0.0, 0.2, 0.5],
        learning_rate=[1e-3, 5e-3, 1e-2, 2e-2, 5e-2], weight_decay=[0.0, 1e-5, 1e-4]),
    'DGAGNN': dict(
        n_hidden=[32, 64, 128], n_head=[1, 2], n_bins=[4, 8, 16, 32],
        z=[0.025, 0.05, 0.1, 0.2], dropout=DROPOUT,
        learning_rate=[1e-3, 5e-3, 1e-2], weight_decay=[1e-4, 5e-4, 1e-3]),
    'FRAUDRE': dict(
        embed_dim=[16, 32, 64, 128], hidden_dim=[32, 64, 128], n_layers=[1, 2, 3],
        batch_size=BATCH, learning_rate=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'GDN': dict(
        emb_size=[32, 64, 128], topk=[5, 10, 20], beta=[0.05, 0.1, 0.5],
        batch_size=BATCH, learning_rate=[1e-3, 5e-3, 1e-2], weight_decay=[1e-4, 1e-3, 1e-2]),
    'GGCN': dict(
        nlayers=[1, 2, 3], hidden=HIDDEN, dropout=[0.0, 0.2, 0.5],
        decay_rate=[0.5, 1.0, 1.5], exponent=[2.0, 3.0],
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[0.0, 1e-5, 1e-4, 1e-3, 4e-3]),
    'GHRN': dict(
        hid_dim=HIDDEN, order=[2, 3, 4, 5],
        del_ratio=[0.01, 0.015, 0.03, 0.05, 0.1],               # GADBench 10^U(-2,-1)
        learning_rate=LR, weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'GLHN': dict(
        hid_dim=HIDDEN, n_layers=[1, 2, 3], tau=[0.05, 0.1, 0.5],
        gamma1=[0.2, 0.4, 0.6, 0.8, 1.0], gamma2=[0.2, 0.4, 0.6, 0.8, 1.0],  # GLHAD sensitivity range
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[1e-5, 1e-4, 1e-3, 1e-2]),
    'GraphConsis': dict(
        dim_1=[32, 64, 128, 200], dropout=DROPOUT, batch_size=BATCH,
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[0.0, 1e-5, 1e-4]),
    'H2FDetector': dict(
        intra_dim=[8, 16, 32], head=[2, 4], n_layer=[1, 2], dropout=[0.0, 0.1, 0.2],
        gamma1=[0.4, 0.8, 1.2, 1.6, 2.0], gamma2=[0.4, 0.8, 1.2, 1.6, 2.0],
        learning_rate=[1e-2, 5e-2, 1e-1], weight_decay=[5e-5, 1e-4, 5e-4]),
    'H2IDE': dict(
        n_factors=[4, 8, 12], factor_dim=[8, 16], alpha1=[0.8, 1.2],
        alpha2=[0.001, 0.01, 0.1], dropout=[0.0, 0.1, 0.2], batch_size=[256, 1024],
        learning_rate=[1e-3, 5e-3, 1e-2], weight_decay=[5e-5, 1e-4, 5e-4]),
    'ImGSL': dict(
        emb_size=[32, 64, 128], beta=[0.3, 0.5, 0.7], epsilon=[0.0, 0.25, 0.5],
        tau_minus=[0.1, 0.2, 0.3], batch_size=[256, 512, 1024],  # IGSL Sec. 5.8 ranges
        learning_rate=[1e-3, 5e-3, 1e-2], weight_decay=[1e-4, 1e-3, 1e-2]),
    'PCGNN': dict(
        emb_size=[32, 64, 128], rho=[0.3, 0.5, 0.7], alpha=[1.0, 2.0, 4.0],
        thres=[0.4, 0.5], batch_size=BATCH,
        learning_rate=[1e-3, 5e-3, 1e-2, 5e-2], weight_decay=[1e-4, 5e-4, 1e-3]),
    'PMP': dict(                                                # PMP App. D.4 search ranges
        hid_dim=[32, 64, 128, 256, 512], dropout=[0.0, 0.2, 0.4, 0.6, 0.8],
        batch_size=[64, 128, 256, 512, 1024],
        learning_rate=[1e-3, 5e-3, 1e-2], weight_decay=[0.0, 5e-5, 1e-4]),
    'SEC_GFD': dict(
        hid_dim=HIDDEN, order=[2, 3, 4, 5],                     # Beta-wavelet order C
        high_order=[1, 2, 3], lemda=[0.1, 0.2, 0.5, 1.0],       # high-pass filters, env. loss weight
        learning_rate=LR, weight_decay=[0.0, 1e-5, 1e-4, 1e-3]),
    'SparseGAD': dict(
        hidden=[16, 32, 64], dropout=[0.0, 0.1, 0.2],           # SparseGAD grid
        learning_rate=[1e-3, 5e-3, 1e-2], k=[5, 10, 20],        # SparseGAD grid
        dropout_adj=[0.1, 0.25, 0.5], weight_decay=[1e-4, 5e-4, 1e-3]),
    'SplitGNN': dict(
        learning_rate=[1e-3, 1e-2, 1e-1],                       # SplitGNN per-dataset values
        C=[2, 3, 4], K=[1, 2], gamma=[0.4, 0.6, 0.8], intra_dim=[8, 16],
        dropout=[0.0, 0.1, 0.2], weight_decay=[5e-5, 1e-4, 5e-4]),
    'TAM': dict(
        embedding_dim=[64, 128, 256], cutting=[3, 4, 5, 7],     # TAM Sec. 4.2 K study
        n_tree=[1, 2, 3], lamda=[0.0, 0.5, 1.0],                # TAM App. C.1
        learning_rate=[1e-5, 5e-5, 1e-4]),
}

# Narrower ranges where a dataset needs them (memory / feature count).
DATASET_OVERRIDES = {
    'tsocial': {
        'AMNet': dict(hidden=[16, 32]),
        'BSL': dict(hid_dim=[32, 48, 64], batch_size=[256, 512]),
        'BWGNN': dict(hid_dim=[10, 16, 32]),
        'CAREGNN': dict(emb_size=[16, 32, 64]),
        'CGNN': dict(emb_size=[32, 64], batch_size=[1024, 4096]),
        'CIEGAD': dict(hidden=[8, 16, 32], order=[2, 4, 6]),
        'COFRAUD': dict(hidden=[16, 32], n_heads=[2, 4]),
        'DGAGNN': dict(n_hidden=[16, 32, 64]),
        'GDN': dict(topk=[2, 3, 5], emb_size=[32, 64]),           # 10 features
        'GGCN': dict(hidden=[10, 16, 32], nlayers=[1, 2]),
        'GHRN': dict(hid_dim=[10, 16, 32]),
        'GLHN': dict(hid_dim=[10, 16, 32]),
        'GraphConsis': dict(dim_1=[32, 64, 128], batch_size=[256, 512]),
        'H2FDetector': dict(intra_dim=[8, 16], head=[2, 4]),
        'ImGSL': dict(emb_size=[32, 64]),
        'SparseGAD': dict(hidden=[16, 32]),
        'TAM': dict(embedding_dim=[64, 128], cutting=[3, 4, 5], n_tree=[1, 2]),
    },
    'tfinance': {
        'GDN': dict(topk=[2, 3, 5]),                               # 10 features
    },
}


def search_space(model, dataset):
    space = dict(SPACES[model])
    space.update(DATASET_OVERRIDES.get(dataset, {}).get(model, {}))
    return space
