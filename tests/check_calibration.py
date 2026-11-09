import polars as pl
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


df = pl.read_csv("results/bayesian_predictions.csv")

y = df["y_true"].to_numpy()
probs = df["prob_bad"].to_numpy()

fraction_positive, mean_predicted = calibration_curve(
    y, probs, n_bins=10, strategy="quantile"
)

brier = brier_score_loss(y, probs)

print("Brier score:", brier)

plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.plot(mean_predicted, fraction_positive, "o-", label="CategoricalNB")

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of actual Bad clients")
plt.title("CategoricalNB - Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("results/bayesian_calibration.png", dpi=200)
plt.show()