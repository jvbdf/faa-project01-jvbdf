import numpy as np
import pyBlindOpt
from src.evaluation import mean_cost
from sklearn.base import BaseEstimator, ClassifierMixin


def evolutionary_fitness(weights, X, y):
    scores = X @ weights[:-1] + weights[-1]
    predictions = (scores >= 0).astype(int)
    return mean_cost(y, predictions)


def optimize_weights(X, y, n_pop=30, n_iter=30, seed=106078):
    bounds = np.array([[-2.0, 2.0]] * (X.shape[1] + 1))

    def objective(weights):
        return evolutionary_fitness(weights, X, y)

    best_weights, best_cost = pyBlindOpt.differential_evolution(
        objective, bounds,
        variant="rand/1/bin",
        policy="fixed",
        F=0.7,
        cr=0.9,
        n_pop=n_pop,
        n_iter=n_iter,
        seed=seed
    )

    return best_weights, best_cost

class EvolutionaryClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_pop=30, n_iter=30, seed=106078):
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        self.weights_, self.cost_ = optimize_weights(X, y, self.n_pop, self.n_iter, self.seed)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        scores = X @ self.weights_[:-1] + self.weights_[-1]
        return (scores >= 0).astype(int)