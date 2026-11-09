from src.results import save_summary


results = [
    {"model": "Decision Tree", "mean_cost": 0.621, "std_cost": 0.08496,
     "macro_f1": 0.54515, "bad_precision": 0.38683, "bad_recall": 0.86, "accuracy": 0.547},

    {"model": "MLP", "mean_cost": 0.602, "std_cost": 0.04804,
     "macro_f1": 0.54693, "bad_precision": 0.39181, "bad_recall": 0.87333, "accuracy": 0.55},

    {"model": "Evolutionary", "mean_cost": 0.705, "std_cost": 0.09987,
     "macro_f1": 0.45910, "bad_precision": 0.34719, "bad_recall": 0.85667, "accuracy": 0.467},

    {"model": "CategoricalNB", "mean_cost": 0.547, "std_cost": 0.04645,
     "macro_f1": 0.55233, "bad_precision": 0.39519, "bad_recall": 0.91667, "accuracy": 0.553},

    {"model": "KNN", "mean_cost": 0.575, "std_cost": 0.14577,
     "macro_f1": 0.57493, "bad_precision": 0.40460, "bad_recall": 0.87333, "accuracy": 0.577}
]

save_summary(results)
print("Results saved!")