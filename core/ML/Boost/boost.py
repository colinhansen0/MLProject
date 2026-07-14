"""From-scratch gradient boosting, objective-agnostic via a pluggable loss.

Builds an additive model as a sequence of shallow regression trees, each fit to
the negative gradient (the pseudo-residuals) of a loss at the current
predictions, then added to a running score at a small learning rate. The same
loop serves regression and classification; only the loss swaps. Validated
against sklearn's GradientBoostingRegressor and GradientBoostingClassifier on
the same feature matrix.

Pieces:
    - sigmoid:       squash an unbounded score into a (0, 1) probability
    - Loss:          the objective-specific trio, init (starting score),
                     neg_gradient (pseudo-residuals from y and current F),
                     output (turn the final score into a prediction)
    - SQUARED_ERROR: regression, mean start, y - F residual, identity output
    - LOG_LOSS:      classification, log-odds start, y - sigmoid(F) residual,
                     sigmoid output
    - Boost.fit:     start at the loss's init value, then for n_estimators rounds
                     fit a regression tree to the current pseudo-residuals and add
                     learning_rate * its predictions to the running score
    - Boost.predict: replay the stored trees from the init value, then apply the
                     loss's output transform

Design notes:
    - The base learner is ALWAYS a regression tree (variance + mean), even for
      classification — the task lives in the loss, never in the tree.
    - Classification accumulates in log-odds space; F is a score, not a
      probability, and the sigmoid converts it only at the output.
    - Leaves use the plain mean of the residuals (vanilla gradient boosting), not
      the Newton-refined leaf values sklearn's classifier uses, so the
      classification fit trails sklearn's slightly.
    - Depends on: core.tree (DecisionTree, REGRESSION).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

from core.ML.tree import DecisionTree, REGRESSION

class Boost:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=None, random_state=None, loss=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss if loss is not None else SQUARED_ERROR
        self.trees = []

    def fit(self, X, y) -> Boost:
        self.trees = []
        self.initial_prediction = self.loss.init(y)
        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = self.loss.neg_gradient(y, F)
            tree = DecisionTree(task=REGRESSION, max_depth=self.max_depth).fit(X, residuals)
            self.trees.append(tree)
            F = F + self.learning_rate * tree.predict(X)

        return self

    def predict(self, X) -> np.ndarray:
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return self.loss.output(F)
    
def sigmoid(F):
    return 1 / (1 + np.exp(-F))

@dataclass(frozen=True)
class Loss:
    init: Callable
    neg_gradient: Callable
    output: Callable

SQUARED_ERROR = Loss(
    init=lambda y: y.mean(),
    neg_gradient=lambda y, F: y - F,
    output=lambda F: F,
)

LOG_LOSS = Loss(
    init=lambda y: np.log(y.mean() / (1 - y.mean())),
    neg_gradient=lambda y, F: y - sigmoid(F),
    output=lambda F: sigmoid(F),
)