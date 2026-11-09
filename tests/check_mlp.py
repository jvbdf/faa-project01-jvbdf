from src.data import load_dataset, prepare_target
from src.experiment import get_outer_folds, get_inner_folds
from src.models import get_mlp_model


df = prepare_target(load_dataset())
X = df.drop("class")
y = df["class"].to_numpy()

tr, te = get_outer_folds(y)[0]
itr, iv = get_inner_folds(y[tr])[0]

model = get_mlp_model()
model.set_params(model__max_iter=800)
model.fit(X[tr[itr]], y[tr[itr]])

loss = model.named_steps["model"].loss_curve_

print("Initial loss:", loss[0])
print("Loss at iteration 250:", loss[249])
print("Final loss:", loss[-1])
print("Last 10 iterations:", loss[-10:])