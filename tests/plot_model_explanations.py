
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyBlindOpt

from matplotlib.patches import Patch
from sklearn.tree import plot_tree

from src.data import load_dataset, prepare_target, CATEGORICAL_FEATURES
from src.experiment import get_outer_folds, get_inner_folds
from src.models import get_tree_model, get_bayesian_model, get_knn_model
from src.preprocessing import get_scaled_preprocessor
from src.evolutionary import evolutionary_fitness


figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

df = prepare_target(load_dataset())
X = df.drop("class")
y = df["class"].to_numpy()

outer_train, _ = get_outer_folds(y)[0]
inner_train, inner_val = get_inner_folds(y[outer_train])[0]

X_train = X[outer_train[inner_train]]
y_train = y[outer_train[inner_train]]

X_val = X[outer_train[inner_val]]
y_val = y[outer_train[inner_val]]


def plot_decision_tree():
    model = get_tree_model(max_depth=5, min_samples_leaf=15)
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessing"]
    tree = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(16, 8))

    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["Good", "Bad"],
        filled=True,
        rounded=True,
        max_depth=2,
        fontsize=10,
        ax=ax
    )

    ax.set_title("Decision Tree - First Three Levels")
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_tree_structure.png", dpi=200)
    plt.close()


def plot_mlp_architecture():
    preprocessor = get_scaled_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)

    n_inputs = X_transformed.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    input_y = np.linspace(1, 5, 5)
    hidden_y = np.linspace(0.5, 5.5, 8)

    for y1 in input_y:
        for y2 in hidden_y:
            ax.plot([1, 3], [y1, y2], color="lightgray", linewidth=0.7)

    for y1 in hidden_y:
        ax.plot([3, 5], [y1, 3], color="lightgray", linewidth=0.7)

    ax.scatter([1] * 5, input_y, s=450, color="#dbeafe",
               edgecolors="#2563eb", zorder=3)

    ax.scatter([3] * 8, hidden_y, s=450, color="#dcfce7",
               edgecolors="#16a34a", zorder=3)

    ax.scatter([5], [3], s=550, color="#fee2e2",
               edgecolors="#dc2626", zorder=3)

    ax.text(1, 6, f"Input Layer\n{n_inputs} features", ha="center")
    ax.text(3, 6, "Hidden Layer\n8 neurons (ReLU)", ha="center")
    ax.text(5, 6, "Output Layer\n1 neuron", ha="center")

    ax.text(1.2, 3, "...", fontsize=18)
    ax.text(5.35, 3, "P(Bad)", va="center")
    ax.text(1, 0.35, "Only 5 input nodes shown", ha="center", fontsize=9)

    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(figures_dir / "mlp_architecture.png", dpi=200)
    plt.close()


def plot_evolutionary_convergence():
    preprocessor = get_scaled_preprocessor()
    X_scaled = preprocessor.fit_transform(X_train)

    bounds = np.array([[-2.0, 2.0]] * (X_scaled.shape[1] + 1))

    def objective(weights):
        return evolutionary_fitness(weights, X_scaled, y_train)

    weights, cost, history = pyBlindOpt.differential_evolution(
        objective,
        bounds,
        variant="rand/1/bin",
        policy="fixed",
        F=0.7,
        cr=0.9,
        n_pop=30,
        n_iter=30,
        seed=106078,
        debug=True
    )

    best, average, worst = history
    generations = np.arange(1, len(best) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(generations, best, label="Best fitness")
    plt.plot(generations, average, label="Mean fitness")

    plt.xlabel("Generation")
    plt.ylabel("Training classification cost")
    plt.title("Differential Evolution - Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(figures_dir / "evolutionary_convergence.png", dpi=200)
    plt.close()

    print("Evolutionary best training cost:", round(cost, 4))


def plot_bayesian_probabilities():
    model = get_bayesian_model(alpha=2.0, n_bins=5)
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessing"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    bayesian = model.named_steps["model"]

    idx = CATEGORICAL_FEATURES.index("checking_status")

    categories = encoder.categories_[idx]
    probabilities = np.exp(
        bayesian.feature_log_prob_[idx][:, 1:len(categories) + 1]
    )

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(x - width / 2, probabilities[0], width, label="Good")
    plt.bar(x + width / 2, probabilities[1], width, label="Bad")

    plt.xticks(x, categories, rotation=20, ha="right")
    plt.xlabel("Checking Account Status")
    plt.ylabel("Conditional Probability")
    plt.title("CategoricalNB - Conditional Probabilities")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir / "bayesian_conditional_probabilities.png",
        dpi=200
    )
    plt.close()


def plot_knn_neighbors():
    model = get_knn_model(n_neighbors=21, weights="distance", p=1)
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessing"]
    knn = model.named_steps["model"]

    X_transformed = preprocessor.transform(X_val)
    distances, indices = knn.kneighbors(X_transformed)

    # Choose an example with neighbors from both classes
    bad_counts = y_train[indices].sum(axis=1)
    sample_idx = np.argmin(np.abs(bad_counts - 10.5))

    neighbor_distances = distances[sample_idx]
    neighbor_classes = y_train[indices[sample_idx]]

    colors = [
        "#4C78A8" if c == 0 else "#F58518"
        for c in neighbor_classes
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(range(1, 22), neighbor_distances, color=colors)

    ax.set_xlabel("Manhattan Distance")
    ax.set_ylabel("Neighbor Rank")
    ax.set_title("KNN - 21 Nearest Neighbors")
    ax.set_yticks(range(1, 22))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Patch(color="#4C78A8", label="Good"),
        Patch(color="#F58518", label="Bad")
    ])

    plt.tight_layout()
    plt.savefig(figures_dir / "knn_neighbors.png", dpi=200)
    plt.close()

    sample = X_val[sample_idx:sample_idx + 1]
    prob_bad = model.predict_proba(sample)[0, 1]

    print("KNN validation sample:", sample_idx)
    print("Actual class:", "Bad" if y_val[sample_idx] == 1 else "Good")
    print("Good neighbors:", int((neighbor_classes == 0).sum()))
    print("Bad neighbors:", int((neighbor_classes == 1).sum()))
    print("Distance-weighted vote for Bad:", round(prob_bad, 4))


plot_decision_tree()
plot_mlp_architecture()
plot_evolutionary_convergence()
plot_bayesian_probabilities()
plot_knn_neighbors()

print("\nAll model figures saved!")
