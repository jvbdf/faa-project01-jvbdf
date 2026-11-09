import numpy as np
import polars as pl
from sklearn.naive_bayes import CategoricalNB

from src.data import load_dataset, prepare_target, CATEGORICAL_FEATURES
from src.experiment import get_outer_folds, get_inner_folds
from src.preprocessing import get_bayesian_preprocessor


def test_bayesian_unknown_category():
    df = prepare_target(load_dataset())
    X = df.drop("class")
    y = df["class"].to_numpy()

    tr, te = get_outer_folds(y)[0]
    itr, iv = get_inner_folds(y[tr])[0]

    X_train = X[tr[itr]]
    X_val = X[tr[iv]].with_columns(pl.lit("unknown_housing").alias("housing"))

    preprocessor = get_bayesian_preprocessor()
    a = preprocessor.fit_transform(X_train)
    b = preprocessor.transform(X_val)

    min_categories = a.max(axis=0).astype(int) + 1
    model = CategoricalNB(alpha=0.5, min_categories=min_categories)
    model.fit(a, y[tr[itr]])

    housing_idx = CATEGORICAL_FEATURES.index("housing")

    assert np.all(b[:, housing_idx] == 0)
    assert model.predict(b).shape == (200,)
    assert model.predict_proba(b).shape == (200, 2)