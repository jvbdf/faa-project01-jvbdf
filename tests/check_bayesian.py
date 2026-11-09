import numpy as np
from sklearn.metrics import confusion_matrix

from src.data import load_dataset, prepare_target
from src.config import BAYESIAN_PARAMS
from src.models import get_bayesian_model
from src.experiment import get_outer_folds, get_inner_folds, tune_model
from pathlib import Path
import polars as pl

df = prepare_target(load_dataset())
X = df.drop("class")
y = df["class"].to_numpy()

predictions = np.zeros(len(y), dtype=int)
probabilities = np.zeros(len(y))
thresholds = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(get_outer_folds(y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    inner_folds = get_inner_folds(y_train)

    params, threshold, inner_cost = tune_model(
        get_bayesian_model, BAYESIAN_PARAMS, X_train, y_train, inner_folds
    )

    model = get_bayesian_model(**params)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    probabilities[test_idx] = probs
    thresholds[test_idx] = threshold
    predictions[test_idx] = (probs >= threshold).astype(int)

    print(f"Fold {fold}: threshold={threshold:.2f}, params={params}")


cm = confusion_matrix(y, predictions, labels=[0, 1])

print("\nConfusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nTotal cost:", fp + 5 * fn)
print("Mean cost:", (fp + 5 * fn) / len(y))

Path("results").mkdir(exist_ok=True)

results = pl.DataFrame({
    "y_true": y,
    "y_pred": predictions,
    "prob_bad": probabilities,
    "threshold": thresholds
})

results.write_csv("results/bayesian_predictions.csv")

print("\nPredictions saved!")