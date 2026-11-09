import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results")
REFERENCE = "CategoricalNB"
MODELS = ["KNN", "MLP", "Decision Tree", "Evolutionary"]


def read_csv(filename):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compare_folds():
    costs = defaultdict(dict)
    for row in read_csv("fold_results.csv"):
        costs[row["model"]][int(row["fold"])] = float(row["mean_cost"])

    reference_folds = costs[REFERENCE]
    summary = []

    print("Paired fold differences (learner - CategoricalNB):")
    for model in MODELS:
        assert costs[model].keys() == reference_folds.keys()
        differences = np.array([
            costs[model][fold] - reference_folds[fold]
            for fold in sorted(reference_folds)
        ])

        row = {
            "learner": model,
            "mean_difference": float(differences.mean()),
            "sample_std": float(differences.std(ddof=1)),
            "folds_better_than_nb": int((differences < 0).sum()),
            "folds_worse_than_nb": int((differences > 0).sum()),
        }
        for fold, value in zip(sorted(reference_folds), differences):
            row[f"fold_{fold}"] = float(value)
        summary.append(row)
        print(f"{model:14s} {differences.round(3)}  mean={row['mean_difference']:.3f}  SD={row['sample_std']:.3f}")

    with (RESULTS / "paired_fold_differences.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def compare_predictions():
    predictions = defaultdict(dict)
    for row in read_csv("predictions.csv"):
        model = row["model"]
        sample_id = int(row["sample_id"])
        assert sample_id not in predictions[model], f"Repeated test observation: {model}, {sample_id}"
        predictions[model][sample_id] = (
            int(row["fold"]), int(row["y_true"]), int(row["y_pred"])
        )

    nb = predictions[REFERENCE]
    knn = predictions["KNN"]
    assert len(nb) == 1000 and nb.keys() == knn.keys()

    counts = defaultdict(int)
    for sample_id, (fold, actual, nb_prediction) in nb.items():
        knn_fold, knn_actual, knn_prediction = knn[sample_id]
        assert (fold, actual) == (knn_fold, knn_actual)
        counts[(actual, nb_prediction == actual, knn_prediction == actual)] += 1

    print("\nCategoricalNB vs KNN, per observation:")
    for actual, label in [(0, "Good"), (1, "Bad")]:
        nb_correct_knn_wrong = counts[(actual, True, False)]
        nb_wrong_knn_correct = counts[(actual, False, True)]
        print(f"{label}: NB correct / KNN wrong = {nb_correct_knn_wrong}; "
              f"NB wrong / KNN correct = {nb_wrong_knn_correct}")

    different = sum(v for (actual, nb_correct, knn_correct), v in counts.items() if nb_correct != knn_correct)
    print(f"Predictions disagree on {different} observations.")


if __name__ == "__main__":
    compare_folds()
    compare_predictions()