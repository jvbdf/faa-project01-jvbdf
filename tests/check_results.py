import polars as pl

files = [
    "fold_results.csv",
    "model_summary.csv",
    "predictions.csv",
    "baselines.csv"
]

for filename in files:
    df = pl.read_csv(f"results/{filename}")
    print(f"{filename}: {df.height} rows")
    print("Nulls:", df.null_count().sum_horizontal().sum())
    print()

summary = pl.read_csv("results/model_summary.csv")

print(summary.select(
    "model",
    "mean_cost_mean",
    "mean_cost_std"
).sort("mean_cost_mean"))