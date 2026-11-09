from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import CategoricalNB
from src.preprocessing import get_bayesian_preprocessor
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing import get_tree_preprocessor, get_scaled_preprocessor
from src.evolutionary import EvolutionaryClassifier


def get_tree_model(max_depth=3, min_samples_leaf=15):
    return Pipeline([
        ("preprocessing", get_tree_preprocessor()),
        ("model", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion="gini",
            class_weight=None,
            random_state=106078
        ))
    ])


def get_mlp_model(hidden_layer_sizes=(8,), alpha=0.001):
    return Pipeline([
        ("preprocessing", get_scaled_preprocessor()),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=False,
            random_state=106078
        ))
    ])

def get_bayesian_model(alpha=0.5, n_bins=3):
    return Pipeline([
        ("preprocessing", get_bayesian_preprocessor(n_bins)),
        ("model", CategoricalNB(alpha=alpha))
    ])
    
def get_knn_model(n_neighbors=7, weights="uniform", p=1):
    return Pipeline([
        ("preprocessing", get_scaled_preprocessor()),
        ("model", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        ))
    ])

def get_evolutionary_model(n_pop=30, n_iter=30):
    return Pipeline([
        ("preprocessing", get_scaled_preprocessor()),
        ("model", EvolutionaryClassifier(n_pop=n_pop, n_iter=n_iter))
    ])