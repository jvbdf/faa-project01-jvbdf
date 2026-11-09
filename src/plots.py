from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
from src.data import load_dataset, NUMERIC_FEATURES
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.evaluation import mean_cost
from src.data import load_dataset, prepare_target
from src.experiment import get_outer_folds, get_inner_folds
from src.models import get_tree_model
from src.evaluation import mean_cost


FIGURES_DIR = Path("results/figures")


def plot_model_comparison():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv("results/model_summary.csv")
    df = df.sort("mean_cost_mean")

    models = df["model"].to_list()
    means = df["mean_cost_mean"].to_numpy()
    stds = df["mean_cost_std"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(models, means, yerr=stds, capsize=5)

    plt.ylabel("Mean classification cost")
    plt.title("Comparison of the Five Classes of Learners")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close()
    
def plot_target_distribution():
    df = load_dataset()
    counts = df["class"].value_counts().sort("class")

    plt.figure(figsize=(6, 4))
    plt.bar(counts["class"].to_list(), counts["count"].to_list())

    plt.title("Target Distribution")
    plt.xlabel("Credit Risk")
    plt.ylabel("Number of Clients")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()


def plot_numeric_distributions():
    df = load_dataset()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for ax, feature in zip(axes.flat, NUMERIC_FEATURES):
        ax.hist(df[feature].to_numpy(), bins=20)
        ax.set_title(feature)

    axes.flat[-1].axis("off")

    fig.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=200)
    plt.close()


def plot_features_by_class():
    df = load_dataset()
    features = ["duration", "credit_amount", "age"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, feature in zip(axes, features):
        good = df.filter(pl.col("class") == "good")[feature].to_numpy()
        bad = df.filter(pl.col("class") == "bad")[feature].to_numpy()

        ax.boxplot([good, bad], tick_labels=["Good", "Bad"])
        ax.set_title(feature)

    fig.suptitle("Numeric Features by Credit Risk")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "features_by_class.png", dpi=200)
    plt.close()
    
def plot_confusion_matrices():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv("results/predictions.csv")

    models = [
        "Decision Tree",
        "MLP",
        "Evolutionary",
        "CategoricalNB",
        "KNN"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for ax, model in zip(axes.flat, models):
        data = df.filter(pl.col("model") == model)

        cm = confusion_matrix(
            data["y_true"].to_numpy(),
            data["y_pred"].to_numpy(),
            labels=[0, 1]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Good", "Bad"]
        )

        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(model)

    axes.flat[-1].axis("off")

    fig.suptitle("Confusion Matrices - Five Classes of Learners")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    plt.close()
    
def plot_model_metrics():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv("results/model_summary.csv")

    models = ["Decision Tree", "MLP", "Evolutionary", "CategoricalNB", "KNN"]

    metrics = {
        "mean_cost": "Mean Cost (lower is better)",
        "macro_f1": "Macro-F1",
        "bad_recall": "Recall (Bad)",
        "bad_precision": "Precision (Bad)",
        "accuracy": "Accuracy"
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (metric, title) in zip(axes.flat, metrics.items()):
        values = [df.filter(pl.col("model") == m)[f"{metric}_mean"].item() for m in models]
        stds = [df.filter(pl.col("model") == m)[f"{metric}_std"].item() for m in models]

        ax.bar(models, values, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)

        if metric != "mean_cost":
            ax.set_ylim(0, 1)

    axes.flat[-1].axis("off")

    fig.suptitle("Comparison of the Five Classes of Learners")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics.png", dpi=200)
    plt.close()
    
def plot_threshold_cost():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv("results/predictions.csv")
    df = df.filter(pl.col("model") == "CategoricalNB")

    y = df["y_true"].to_numpy()
    probs = df["prob_bad"].to_numpy()
    predictions = df["y_pred"].to_numpy()

    thresholds = np.linspace(0, 1, 101)
    costs = []

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        costs.append(mean_cost(y, y_pred))

    actual_cost = mean_cost(y, predictions)

    plt.figure(figsize=(9, 5))

    plt.plot(thresholds, costs, label="Fixed threshold")
    plt.axhline(actual_cost, linestyle="--",
                label=f"Nested CV cost = {actual_cost:.3f}")
    plt.axvline(1/6, linestyle=":",
                label="Theoretical threshold = 0.167")

    plt.xlabel("Classification threshold")
    plt.ylabel("Mean classification cost")
    plt.title("CategoricalNB - Threshold Cost Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "threshold_cost.png", dpi=200)
    plt.close()

def plot_tree_validation_curve():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_target(load_dataset())
    X = df.drop("class")
    y = df["class"].to_numpy()

    train_idx, test_idx = get_outer_folds(y)[0]
    X_train, y_train = X[train_idx], y[train_idx]

    folds = get_inner_folds(y_train)
    depths = [2, 3, 4, 5, 6, 8, 10, 12]

    train_means = []
    val_means = []
    val_stds = []

    for depth in depths:
        train_costs = []
        val_costs = []

        for tr, val in folds:
            model = get_tree_model(max_depth=depth, min_samples_leaf=15)
            model.fit(X_train[tr], y_train[tr])

            train_probs = model.predict_proba(X_train[tr])[:, 1]
            val_probs = model.predict_proba(X_train[val])[:, 1]

            train_pred = (train_probs >= 1/6).astype(int)
            val_pred = (val_probs >= 1/6).astype(int)

            train_costs.append(mean_cost(y_train[tr], train_pred))
            val_costs.append(mean_cost(y_train[val], val_pred))

        train_means.append(np.mean(train_costs))
        val_means.append(np.mean(val_costs))
        val_stds.append(np.std(val_costs, ddof=1))

    plt.figure(figsize=(9, 5))

    plt.plot(depths, train_means, "o-", label="Training cost")
    plt.errorbar(depths, val_means, yerr=val_stds,
                 fmt="o-", capsize=4, label="Validation cost")

    plt.xlabel("Maximum tree depth")
    plt.ylabel("Mean classification cost")
    plt.title("Decision Tree - Validation Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "tree_validation_curve.png", dpi=200)
    plt.close()