"""From-scratch CART decision tree, objective-agnostic via a pluggable task.

Grows a binary tree by recursively choosing, at each node, the split that most
reduces a task-defined impurity; each leaf emits a task-defined value over the
rows that reach it. The same structure serves both classification and regression,
only the impurity and leaf value swap (gini + majority class, or variance +
mean). Validated against sklearn's DecisionTreeClassifier and
DecisionTreeRegressor on the same feature matrix.

Pieces:
    - gini(y) / variance(y): node impurity for classification / regression
    - best_split:  scan features, pick the largest impurity drop; with
                   max_features, consider only a random subset of features
    - fit / grow:  recurse until a stopping rule (max depth, pure node, or no
                   useful split)
    - predict:     route each row down to a leaf and return its value

Design notes:
    - Features are binary fingerprint bits, so each split is just "bit off vs on"
      (one candidate threshold per feature).
    - Impurity + leaf value are the only objective-specific parts — bundled in a
      task so the tree core stays agnostic.
    - This single tree is the unit that forest.py composes.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

@dataclass
class Node:
    """Class of node, includes the feature split on, node for left or right, 
    and if it is a leaf, the output prediction
    
    is_leaf function returns true if the node is a leaf"""
    feature: int | None = None    # split bit for an internal node; None on a leaf
    left: Node | None = None    # child Node for bit 0
    right: Node | None = None    # child Node for bit 1
    prediction: int | None = None    # class to output on a leaf; None internally

    def is_leaf(self) -> bool:
        return self.prediction is not None

class DecisionTree:
    """CART-style binary decision tree, objective-agnostic via a pluggable task.

    The same tree structure serves classification or regression; only the task
    (impurity + leaf value) swaps.

    max_features: if set, each split considers a random subset of this many
                  features (the randomness a forest relies on); None scans all
                  features for a plain, deterministic tree.
    max_depth:    if set, a node that reaches this depth becomes a leaf instead
                  of splitting further, capping how deep the tree grows; None
                  lets it grow until another stopping rule fires.
    random_state: seed for the per-split feature sampling, for reproducible runs.
    task:         the objective-specific impurity + leaf value (defaults to
                  CLASSIFICATION).
    """
        
    def __init__(self, max_features=None, random_state=None, task=None, max_depth=None):
        self.root = None
        self.max_features = max_features
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
        self.task = task if task is not None else CLASSIFICATION

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        depth = 0
        self.root = self._grow(X, y, depth)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """4 cases, max depth reached, pure split, no best split, and a case where there is a good split (recursive),
        Utilized in fit to create a single algorithmically generated gini/variance minimizing tree
        """

        #Reach max_depth case
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=self.task.leaf_value(y))

        #Pure split case (100% one group)
        if len(np.unique(y)) == 1:
            return Node(prediction=self.task.leaf_value(y))

        #best split call, returns best feature to split by and the minimization gain
        feature, gain = best_split(X, y, self.task.impurity, self.max_features, self.rng)

        #No split is the best case
        if feature is None:
            return Node(prediction=self.task.leaf_value(y))
        
        #Recursive Case
        mask = X[:, feature] == 0
        X_left,  y_left  = X[mask],  y[mask]
        X_right, y_right = X[~mask], y[~mask]
        left_child  = self._grow(X_left,  y_left, depth + 1)
        right_child = self._grow(X_right, y_right, depth + 1)
        return Node(feature=feature, left=left_child, right=right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for row in X:
            node = self.root
            while not node.is_leaf():
                if row[node.feature] == 0:
                    node=node.left
                else:
                    node=node.right
            preds.append(node.prediction)
        return np.array(preds)
    
def gini(y: np.ndarray) -> float:
    """Gini impurity of a label array: 1 - sum(p_i^2) over the class proportions p_i.

    Measures how mixed the labels are, 0 when the group is pure (all one class),
    rising to a maximum (0.5 for two equally mixed classes) at a perfect 50/50 split.

    For use in greedy minimization.
    """
    counts = np.bincount(y)      # count of each class; index IS the label (counts[0] = #0s, counts[1] = #1s)
    props = counts / len(y)      # proportions: p_i = fraction of the group in each class
    sqsum = (props ** 2).sum()   # the sum(p_i^2) term
    return 1 - sqsum             # Gini impurity = 1 - sum(p_i^2)

def variance(y: np.ndarray) -> float:
    return float(np.var(y))

def best_split(
    X: np.ndarray,
    y: np.ndarray,
    impurity: Callable[[np.ndarray], float] = gini,
    max_features: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int | None, float]:
    best_feature = None
    best_gain = 0.0
    parent_impurity = impurity(y)

    features = range(X.shape[1])
    if max_features is not None:
        features = rng.choice(X.shape[1], size=max_features, replace=False)

    for j in features:
        left  = y[X[:, j] == 0]
        right = y[X[:, j] == 1]
        if len(left) == 0 or len(right) == 0:
            continue
        weighted_child = (len(left) / len(X)) * impurity(left) + (len(right) / len(X)) * impurity(right)
        gain = parent_impurity - weighted_child
        if gain > best_gain:
            best_feature, best_gain = j, gain

    return (best_feature, best_gain)

# leaf values: what a leaf predicts from the targets that reach it
def majority_class(y: np.ndarray) -> int:
    return int(np.bincount(y).argmax())

def mean_value(y: np.ndarray) -> float:
    return float(y.mean())

# aggregation: collapse a (n_trees, n_samples) prediction matrix to n_samples
def majority_vote(preds: np.ndarray) -> np.ndarray:
    return np.array([np.bincount(preds[:, i]).argmax() for i in range(preds.shape[1])])

def mean_aggregate(preds: np.ndarray) -> np.ndarray:
    return preds.mean(axis=0)


@dataclass(frozen=True)
class Task:
    """The objective-specific pieces a tree/forest needs; everything else is agnostic.

    Swapping the task is what turns the same machinery from a classifier into a
    regressor — the tree structure, recursion, and splitting never change.

    impurity:   node impurity from its targets, minimized by best_split
                (gini for classification, variance for regression).
    leaf_value: the value a leaf predicts from its targets
                (majority class, or the mean).
    aggregate:  collapse a (n_trees, n_samples) prediction matrix into one
                prediction per sample, for the forest (majority vote per column,
                or the column mean).
    """
    impurity: Callable[[np.ndarray], float]
    leaf_value: Callable[[np.ndarray], int | float]
    aggregate: Callable[[np.ndarray], np.ndarray]


CLASSIFICATION = Task(impurity=gini, leaf_value=majority_class, aggregate=majority_vote)
REGRESSION = Task(impurity=variance, leaf_value=mean_value, aggregate=mean_aggregate)

if __name__ == "__main__":
    print(variance([5,5,5]))
    print(variance([2,4,6]))
