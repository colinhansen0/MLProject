"""From-scratch random forest classifier.

An ensemble of decision trees (tree.py) trained with two sources of randomness,
then aggregated by majority vote. Validated against sklearn's
RandomForestClassifier on the same feature matrix.

What makes it a forest rather than one tree:
    - bagging:             each tree trains on a bootstrap sample (rows drawn with
                           replacement) of the training data
    - feature subsampling: each split considers only a random subset of the
                           features (typically ~sqrt(n_features))

Pieces to build:
    - bootstrap sampling of training rows
    - train N trees, each on its own bootstrap sample
    - aggregate the trees' predictions (majority vote)

Depends on: tree.py.
"""

from __future__ import annotations

import numpy as np

from core.ML.tree import DecisionTree, CLASSIFICATION

class RandomForest:
    """Bagging ensemble of CART-style classification trees.

    Grows n_estimators trees, each on a bootstrap sample of the rows and each
    splitting on a random subset of features, then predicts by majority vote
    across the trees. Validated against sklearn's RandomForestClassifier on the
    same feature matrix.

    n_estimators: number of trees to grow.
    max_features: features each split may consider within a tree; None uses
                  sqrt(n_features). This per-split randomness, together with the
                  bootstrap rows, is what decorrelates the trees.
    random_state: seed for the forest's rng (bootstrap row sampling and each
                  tree's seed), for reproducible runs.
    """
    def __init__(self, n_estimators=100, max_features=None, random_state=None, task=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.task = task if task is not None else CLASSIFICATION
        self.trees = []

    def fit(self, X, y) -> RandomForest:
        self.trees = []                       # fresh, so re-fitting doesn't accumulate

        row_count = len(X)
        feature_count = len(X[0])

        if self.max_features is None:
            mf = int(np.sqrt(feature_count))
        else:
            mf = self.max_features

        for _ in range(self.n_estimators):
            bootstrap = self.rng.choice(row_count, size=row_count, replace=True)   # bootstrap ROWS, full size
            X_boot, y_boot = X[bootstrap], y[bootstrap]

            seed = int(self.rng.integers(2**32))

            tree = DecisionTree(max_features=mf, random_state=seed, task=self.task).fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X) -> np.ndarray:
        preds = np.array([tree.predict(X) for tree in self.trees])   # (n_trees, n_samples)
        return self.task.aggregate(preds)