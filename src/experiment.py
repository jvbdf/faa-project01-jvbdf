from sklearn.model_selection import StratifiedKFold
import numpy as np
from src.config import THRESHOLDS
from src.evaluation import mean_cost, evaluate_model
from sklearn.base import clone
from src.config import EVOLUTIONARY_PARAMS
from src.models import get_evolutionary_model
import time
SEED = 106078

def get_outer_folds(y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(cv.split(range(len(y)), y))


def get_inner_folds(y):
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    return list(cv.split(range(len(y)), y))

def select_threshold(y_true, probabilities):
    costs = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        costs.append(mean_cost(y_true, predictions))

    best_cost = min(costs)
    candidates = [t for t, cost in zip(THRESHOLDS, costs) if cost == best_cost]

    best_threshold = min(candidates, key=lambda t: (abs(t - 1/6), t))

    return best_threshold, best_cost

def get_oof_probabilities(model, X, y, folds):
    probabilities = np.zeros(len(y))

    for train_idx, val_idx in folds:
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])

        bad_idx = list(clf.classes_).index(1)
        probabilities[val_idx] = clf.predict_proba(X[val_idx])[:, bad_idx]

    return probabilities

def tune_model(get_model, params_list, X, y, folds):
    best_params = None
    best_threshold = None
    best_cost = float("inf")

    for params in params_list:
        model = get_model(**params)
        probabilities = get_oof_probabilities(model, X, y, folds)
        threshold, cost = select_threshold(y, probabilities)

        if cost < best_cost:
            best_cost = cost
            best_params = params
            best_threshold = threshold

    return best_params, best_threshold, best_cost

def evaluate_outer_fold(get_model, params_list, X, y, train_idx, test_idx):
    start = time.perf_counter()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    folds = get_inner_folds(y_train)
    params, threshold, inner_cost = tune_model(
        get_model, params_list, X_train, y_train, folds
    )

    model = get_model(**params)
    model.fit(X_train, y_train)

    bad_idx = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, bad_idx]
    predictions = (probabilities >= threshold).astype(int)

    return {
    "params": params,
    "threshold": threshold,
    "inner_cost": inner_cost,
    "test_idx": test_idx,
    "y_true": y_test,
    "y_pred": predictions,
    "prob_bad": probabilities,
    "time_seconds": time.perf_counter() - start,
    **evaluate_model(y_test, predictions)
    }
    
def evaluate_nested_cv(get_model, params_list, X, y):
    results = []

    for i, (train_idx, test_idx) in enumerate(get_outer_folds(y), 1):
        result = evaluate_outer_fold(
            get_model, params_list, X, y, train_idx, test_idx
        )

        result["fold"] = i
        results.append(result)

    return results

def tune_evolutionary(X, y, folds):
    best_params = None
    best_cost = float("inf")

    for params in EVOLUTIONARY_PARAMS:
        costs = []

        for train_idx, val_idx in folds:
            model = get_evolutionary_model(**params)
            model.fit(X[train_idx], y[train_idx])

            predictions = model.predict(X[val_idx])
            costs.append(mean_cost(y[val_idx], predictions))

        cost = np.mean(costs)

        if cost < best_cost:
            best_cost = cost
            best_params = params

    return best_params, best_cost

def evaluate_evolutionary_fold(X, y, train_idx, test_idx):
    start = time.perf_counter()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    folds = get_inner_folds(y_train)
    params, inner_cost = tune_evolutionary(X_train, y_train, folds)

    model = get_evolutionary_model(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
    "params": params,
    "threshold": None,
    "inner_cost": inner_cost,
    "test_idx": test_idx,
    "y_true": y_test,
    "y_pred": predictions,
    "prob_bad": None,
    "time_seconds": time.perf_counter() - start,
    **evaluate_model(y_test, predictions)
    }
    
def evaluate_evolutionary_cv(X, y):
    results = []

    for i, (train_idx, test_idx) in enumerate(get_outer_folds(y), 1):
        result = evaluate_evolutionary_fold(X, y, train_idx, test_idx)

        result["fold"] = i
        results.append(result)

    return results