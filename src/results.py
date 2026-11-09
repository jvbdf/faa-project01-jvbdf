import json
from pathlib import Path

import polars as pl


RESULTS_DIR = Path("results")


def save_csv(filename, rows):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename

    df = pl.DataFrame(rows)

    if path.exists():
        previous = pl.read_csv(path)
        df = pl.concat([previous, df], how="vertical_relaxed")

    df.write_csv(path)


def save_fold_result(model, result, elapsed_time):
    row = {
        "model": model,
        "fold": result["fold"],
        "params": json.dumps(result["params"]),
        "threshold": result.get("threshold"),
        "inner_cost": float(result["inner_cost"]),
        "mean_cost": float(result["mean_cost"]),
        "macro_f1": float(result["macro_f1"]),
        "bad_precision": float(result["bad_precision"]),
        "bad_recall": float(result["bad_recall"]),
        "accuracy": float(result["accuracy"]),
        "time_seconds": elapsed_time
    }

    save_csv("fold_results.csv", [row])


def save_predictions(model, result):
    probabilities = result["prob_bad"]

    rows = []

    for i, idx in enumerate(result["test_idx"]):
        rows.append({
            "model": model,
            "fold": result["fold"],
            "sample_id": int(idx),
            "y_true": int(result["y_true"][i]),
            "y_pred": int(result["y_pred"][i]),
            "prob_bad": None if probabilities is None else float(probabilities[i]),
            "threshold": result.get("threshold")
        })

    save_csv("predictions.csv", rows)


def save_summary(model, results):
    from src.evaluation import summarize_results

    summary = summarize_results(results)

    row = {"model": model}

    for metric, values in summary.items():
        row[f"{metric}_mean"] = float(values["mean"])
        row[f"{metric}_std"] = float(values["std"])

    save_csv("model_summary.csv", [row])
    
def clear_results():
    files = [
        "fold_results.csv",
        "model_summary.csv",
        "predictions.csv",
        "baselines.csv"
    ]

    for filename in files:
        path = RESULTS_DIR / filename

        if path.exists():
            path.unlink()