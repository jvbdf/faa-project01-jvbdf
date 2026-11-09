import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score


def mean_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (fp + 5 * fn) / len(y_true)


def evaluate_baselines(y_true):
    always_good = np.zeros(len(y_true), dtype=int)
    always_bad = np.ones(len(y_true), dtype=int)

    return {
        "always_good": evaluate_model(y_true, always_good),
        "always_bad": evaluate_model(y_true, always_bad)
    }


def evaluate_model(y_true, y_pred):
    return {
        "mean_cost": mean_cost(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "bad_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "bad_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }
    
def summarize_results(results):
    metrics = ["mean_cost", "macro_f1", "bad_precision", "bad_recall", "accuracy"]

    return {
        metric: {
            "mean": np.mean([r[metric] for r in results]),
            "std": np.std([r[metric] for r in results], ddof=1)
        }
        for metric in metrics
    }