import polars as pl

DATASET_PATH = "datasets/german_credit.csv.zst"
TARGET = "class"

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]


def load_dataset(path=DATASET_PATH):
    df = pl.read_csv(path)

    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

    if set(df.columns) != set(expected):
        raise ValueError("Invalid dataset columns")

    if df.null_count().sum_horizontal().item() > 0:
        raise ValueError("Dataset contains missing values")

    if set(df[TARGET].unique().to_list()) != {"good", "bad"}:
        raise ValueError("Invalid target values")

    return df


def prepare_target(df):
    return df.with_columns(pl.col(TARGET).replace({"good": 0, "bad": 1}).cast(pl.Int8)
    )