import polars as pl
import matplotlib.pyplot as plt

df = pl.read_csv("results/bayesian_predictions.csv")

good = df.filter(pl.col("y_true") == 0)["prob_bad"].to_numpy()
bad = df.filter(pl.col("y_true") == 1)["prob_bad"].to_numpy()

plt.hist(good, bins=25, alpha=0.6, density=True, label="Good")
plt.hist(bad, bins=25, alpha=0.6, density=True, label="Bad")

plt.xlabel("Predicted probability of Bad")
plt.ylabel("Density")
plt.title("CategoricalNB - Probability Distribution")
plt.legend()
plt.savefig("results/bayesian_probabilities.png", dpi=200)
plt.show()