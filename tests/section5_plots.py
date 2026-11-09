from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Script intended to be run from the project root.
results = Path('results')
figures = results / 'figures'
figures.mkdir(parents=True, exist_ok=True)
models = ['CategoricalNB', 'KNN', 'MLP', 'Decision Tree', 'Evolutionary']

with (results / 'fold_results.csv').open(newline='', encoding='utf-8') as f:
    folds = list(csv.DictReader(f))
with (results / 'predictions.csv').open(newline='', encoding='utf-8') as f:
    predictions = list(csv.DictReader(f))

# Same outer folds for every learner: show fold-to-fold behavior.
fig, ax = plt.subplots(figsize=(10, 5.5))
for model in models:
    rows = sorted((r for r in folds if r['model'] == model), key=lambda x: int(x['fold']))
    ax.plot([int(r['fold']) for r in rows], [float(r['mean_cost']) for r in rows],
            marker='o', linewidth=2, label=model)
ax.axhline(.7, linestyle='--', linewidth=1.5, color='gray', label='Always Bad baseline (0.700)')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel('Outer fold')
ax.set_ylabel('Mean classification cost (lower is better)')
ax.set_title('Classification Cost by Outer Fold')
ax.set_ylim(.3, .9)
ax.grid(axis='y', alpha=.22)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(figures / 'outer_fold_costs.png', dpi=200)
plt.close(fig)

# Decompose observed mean cost into FP penalty and 5x FN penalty.
fp_components, fn_components = [], []
for model in models:
    rows = [r for r in predictions if r['model'] == model]
    fp = sum(int(r['y_true']) == 0 and int(r['y_pred']) == 1 for r in rows)
    fn = sum(int(r['y_true']) == 1 and int(r['y_pred']) == 0 for r in rows)
    fp_components.append(fp / len(rows))
    fn_components.append(5 * fn / len(rows))
models_with_base = models + ['Always Bad']
fp_components.append(.7)
fn_components.append(0.)

x = np.arange(len(models_with_base))
fig, ax = plt.subplots(figsize=(10.8, 5.7))
bars_fp = ax.bar(x, fp_components, label='False positives: FP / N')
bars_fn = ax.bar(x, fn_components, bottom=fp_components, label='False negatives: 5 × FN / N')
for i, (fp, fn) in enumerate(zip(fp_components, fn_components)):
    ax.text(i, fp + fn + .015, f'{fp + fn:.3f}', ha='center', fontsize=10)
ax.set_xticks(x, models_with_base, rotation=20, ha='right')
ax.set_ylabel('Mean classification cost (lower is better)')
ax.set_title('Classification Cost by Error Type')
ax.set_ylim(0, .85)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(figures / 'error_cost_breakdown.png', dpi=200)
plt.close(fig)

print('Saved:', figures / 'outer_fold_costs.png')
print('Saved:', figures / 'error_cost_breakdown.png')