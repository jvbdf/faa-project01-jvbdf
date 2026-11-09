SEED = 106078

TREE_PARAMS = [
    {"max_depth": 3, "min_samples_leaf": 15},
    {"max_depth": 5, "min_samples_leaf": 15},
    {"max_depth": 5, "min_samples_leaf": 5},
    {"max_depth": 8, "min_samples_leaf": 5}
]

MLP_PARAMS = [
    {"hidden_layer_sizes": (8,), "alpha": 0.001},
    {"hidden_layer_sizes": (8,), "alpha": 0.1},
    {"hidden_layer_sizes": (16,), "alpha": 0.001},
    {"hidden_layer_sizes": (16,), "alpha": 0.1}
]

EVOLUTIONARY_PARAMS = [
    {"n_pop": 30, "n_iter": 30},
    {"n_pop": 40, "n_iter": 22}
]

BAYESIAN_PARAMS = [
    {"alpha": 0.5, "n_bins": 3},
    {"alpha": 2.0, "n_bins": 3},
    {"alpha": 0.5, "n_bins": 5},
    {"alpha": 2.0, "n_bins": 5}
]

KNN_PARAMS = [
    {"n_neighbors": 7, "weights": "uniform", "p": 1},
    {"n_neighbors": 7, "weights": "uniform", "p": 2},
    {"n_neighbors": 7, "weights": "distance", "p": 1},
    {"n_neighbors": 7, "weights": "distance", "p": 2},
    {"n_neighbors": 21, "weights": "uniform", "p": 1},
    {"n_neighbors": 21, "weights": "uniform", "p": 2},
    {"n_neighbors": 21, "weights": "distance", "p": 1},
    {"n_neighbors": 21, "weights": "distance", "p": 2}
]

THRESHOLDS = [i / 100 for i in range(101)]