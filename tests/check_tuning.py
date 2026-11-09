import json
import polars as pl

df = pl.read_csv("results/fold_results.csv")

with open("results/experiment_config.json") as f:
    config = json.load(f)

for model, params in config["hyperparameters"].items():
    data = df.filter(pl.col("model") == model)

    print(f"\n{model}")
    print("Configurations:", len(params))
    print("Total time:", round(data["time_seconds"].sum(), 2), "seconds")

    for row in data.iter_rows(named=True):
        print(
            "Fold", row["fold"],
            "| Params:", row["params"],
            "| Threshold:", row["threshold"]
        )