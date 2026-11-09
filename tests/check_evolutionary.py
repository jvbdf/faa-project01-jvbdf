import numpy as np
import pyBlindOpt

from src.data import load_dataset, prepare_target
from src.experiment import get_outer_folds, get_inner_folds
from src.preprocessing import get_scaled_preprocessor
from src.evolutionary import evolutionary_fitness


def weighted_cross_entropy(weights, X, y):
    scores = X @ weights[:-1] + weights[-1]

    losses = (
        5 * y * np.logaddexp(0, -scores)
        + (1 - y) * np.logaddexp(0, scores)
    )

    return np.mean(losses)


df = prepare_target(load_dataset())
X = df.drop("class")
y = df["class"].to_numpy()

tr, te = get_outer_folds(y)[0]
folds = get_inner_folds(y[tr])

results = {"Original": [], "WBCE": []}

for fold, (itr, iv) in enumerate(folds, 1):
    preprocessor = get_scaled_preprocessor()

    X_train = preprocessor.fit_transform(X[tr[itr]])
    X_val = preprocessor.transform(X[tr[iv]])

    y_train = y[tr[itr]]
    y_val = y[tr[iv]]

    bounds = np.array([[-2.0, 2.0]] * (X_train.shape[1] + 1))

    objectives = {
        "Original": lambda w: evolutionary_fitness(w, X_train, y_train),
        "WBCE": lambda w: weighted_cross_entropy(w, X_train, y_train)
    }

    for name, objective in objectives.items():
        weights, fitness = pyBlindOpt.differential_evolution(
            objective, bounds,
            variant="rand/1/bin",
            policy="fixed",
            F=0.7,
            cr=0.9,
            n_pop=30,
            n_iter=30,
            seed=106078
        )

        cost = evolutionary_fitness(weights, X_val, y_val)
        results[name].append(cost)

        print(f"Fold {fold} - {name}: {cost:.4f}")

print("\nMean validation cost:")
for name, costs in results.items():
    print(name, np.mean(costs))