from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, FunctionTransformer

def get_tree_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES)
    ])


def get_scaled_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES)
    ])

def get_bayesian_preprocessor(n_bins=3):
    categorical = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("shift", FunctionTransformer(lambda x: x + 1))
    ])

    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("continuous", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"), ["duration", "credit_amount", "age"]),
        ("discrete", "passthrough", ["installment_commitment", "residence_since", "existing_credits", "num_dependents"])
    ])