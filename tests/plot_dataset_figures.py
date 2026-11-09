
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from src.data import load_dataset

import numpy as np
from src.data import NUMERIC_FEATURES, prepare_target

figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

df = load_dataset()
overall_rate = df.select((pl.col("class") == "bad").mean()).item() * 100


def plot_rates(ax, feature, title):
    stats = (
        df.group_by(feature)
        .agg(
            pl.len().alias("count"),
            (pl.col("class") == "bad").sum().alias("bad")
        )
        .with_columns(
            (pl.col("bad") / pl.col("count") * 100).alias("rate")
        )
        .sort("rate", descending=True)
    )

    categories = stats[feature].to_list()
    rates = stats["rate"].to_list()
    counts = stats["count"].to_list()

    ax.barh(categories, rates)
    ax.invert_yaxis()

    ax.axvline(overall_rate, linestyle="--", alpha=0.7)

    ax.set_xlim(0, 78)
    ax.set_xlabel("Bad credit rate (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(
            rate + 1, i,
            f"{rate:.1f}% (n={count})",
            va="center",
            fontsize=9
        )


def plot_categorical_risk():
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5.5), layout="constrained"
    )

    plot_rates(axes[0], "checking_status", "Checking account status")
    plot_rates(axes[1], "credit_history", "Credit history")

    fig.suptitle("Bad credit rate by categorical features")
    fig.supxlabel("Dashed line: overall bad credit rate (30%)")

    plt.savefig(figures_dir / "categorical_risk.png", dpi=200)
    plt.close()


plot_categorical_risk()


def plot_numeric_correlation():
    columns = NUMERIC_FEATURES + ["class"]

    numeric_df = prepare_target(df).select(columns)
    correlation = np.corrcoef(numeric_df.to_numpy(), rowvar=False)

    fig, ax = plt.subplots(figsize=(10.7, 8.2))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=38, ha="right")
    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            value = correlation[i, j]
            color = "white" if abs(value) > 0.6 else "black"

            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center", color=color)

    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("German Credit - Numerical Feature Correlations")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation (r)")

    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_correlation.png", dpi=200)
    plt.close()


plot_numeric_correlation()
