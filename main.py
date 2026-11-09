import json
import time
import platform
from pathlib import Path
from importlib.metadata import version

import polars as pl

from src.data import load_dataset, prepare_target
from src.config import (
    SEED, TREE_PARAMS, MLP_PARAMS, EVOLUTIONARY_PARAMS,
    BAYESIAN_PARAMS, KNN_PARAMS, THRESHOLDS
)
from src.models import (
    get_tree_model, get_mlp_model, get_bayesian_model, get_knn_model
)
from src.experiment import (
    get_outer_folds, evaluate_outer_fold, evaluate_evolutionary_fold
)
from src.evaluation import evaluate_baselines
from src.results import (
    clear_results, save_fold_result, save_predictions,
    save_summary, save_csv
)


def main():
    df = prepare_target(load_dataset())
    X = df.drop("class")
    y = df["class"].to_numpy()

    models = {
        "Decision Tree": (get_tree_model, TREE_PARAMS),
        "MLP": (get_mlp_model, MLP_PARAMS),
        "Evolutionary": (None, EVOLUTIONARY_PARAMS),
        "CategoricalNB": (get_bayesian_model, BAYESIAN_PARAMS),
        "KNN": (get_knn_model, KNN_PARAMS)
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    clear_results()

    config = {
        "seed": SEED,
        "dataset": "german_credit.csv.zst",
        "observations": len(y),
        "outer_folds": 5,
        "inner_folds": 4,
        "cost_fp": 1,
        "cost_fn": 5,
        "thresholds": THRESHOLDS,
        "hyperparameters": {name: params for name, (_, params) in models.items()},
        "python_version": platform.python_version(),
        "libraries": {
            name: version(name)
            for name in ["numpy", "scikit-learn", "polars", "pyBlindOpt"]
        }
    }

    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=4)

    outer_folds = get_outer_folds(y)
    start = time.perf_counter()

    baseline_results = {"always_good": [], "always_bad": []}

    for i, (train_idx, test_idx) in enumerate(outer_folds, 1):
        baselines = evaluate_baselines(y[test_idx])

        for name, metrics in baselines.items():
            baseline_results[name].append(metrics)
            save_csv("baselines.csv", [{
                "model": name,
                "fold": i,
                **metrics
            }])

    for name, (get_model, params_list) in models.items():
        print(f"\n{name}")
        results = []

        for i, (train_idx, test_idx) in enumerate(outer_folds, 1):
            if name == "Evolutionary":
                result = evaluate_evolutionary_fold(X, y, train_idx, test_idx)
            else:
                result = evaluate_outer_fold(
                    get_model, params_list, X, y, train_idx, test_idx
                )

            result["fold"] = i
            results.append(result)

            save_fold_result(name, result, result["time_seconds"])
            save_predictions(name, result)

            print(
                f"Fold {i}/5 | "
                f"Cost: {result['mean_cost']:.4f} | "
                f"Time: {result['time_seconds']:.1f}s",
                flush=True
            )

        save_summary(name, results)

    for name, results in baseline_results.items():
        save_summary(name, results)

    total_time = time.perf_counter() - start
    config["total_time_seconds"] = total_time

    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nExperiment completed in {total_time:.1f} seconds.")


if __name__ == "__main__":
    main()