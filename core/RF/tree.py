"""From-scratch classification decision tree (CART-style).

Grows a binary tree by recursively choosing, at each node, the split that most
reduces label impurity; leaves predict the majority class of the rows that reach
them. Validated against sklearn's DecisionTreeClassifier on the same feature matrix.

Pieces to build:
    - gini(y):     impurity of a label array (0 = pure, 0.5 = even binary mix)
    - best split:  scan features/thresholds, pick the largest impurity drop
    - fit / grow:  recurse until a stopping rule (pure node, min samples, max depth)
    - predict:     route each row down to a leaf and return its class

Design notes:
    - Features are binary fingerprint bits, so each split is just "bit off vs on"
      (one candidate threshold per feature).
    - Classification first; the impurity + leaf logic is the part that later
      generalizes to regression (the objective-agnostic refactor).
    - This single tree is the unit that forest.py composes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Node:
    """Class of node, includes the feature split on, node for left or right, 
    and if it is a lead, the output prediction
    
    is_lead function returns true if the node is a leaf"""
    feature: int | None = None    # split bit for an internal node; None on a leaf
    left: Node | None = None    # child Node for bit 0
    right: Node | None = None    # child Node for bit 1
    prediction: int = None    # class to output on a leaf; None internally

    def is_leaf(self) -> bool:
        return self.prediction is not None

class DecisionTree:
    def __init__(self):
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._grow(X, y)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray) -> Node:
        """3 cases, pure split, no best split, and a case where there is a good split (recursive),
        Utlizied in fit to create a single algorithmically generated gini minimizing tree
        """
        #Pure split case (100% one group)
        if len(np.unique(y)) == 1:
            return Node(prediction=np.bincount(y).argmax())

        #best split call, returns best feature to split by and the minimization gain
        feature, gain = best_split(X, y)

        #No split is the best case
        if feature is None:
            return Node(prediction=np.bincount(y).argmax())
        
        #Recursive Case
        mask = X[:, feature] == 0
        X_left,  y_left  = X[mask],  y[mask]
        X_right, y_right = X[~mask], y[~mask]
        left_child  = self._grow(X_left,  y_left)
        right_child = self._grow(X_right, y_right)
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

def best_split(X:np.ndarray, y:np.ndarray) -> tuple[int | None, float]:
    """(best_feature, best_gain); best_feature is None if no split reduces impurity"""

    best_feature = None 
    best_gain = float(0.0)

    parent_gini=gini(y)

    for j in range(X.shape[1]):
        left  = y[X[:, j] == 0]
        right = y[X[:, j] == 1]

        if len(left) == 0 or len(right) == 0:
            continue

        weighted_child = (len(left) / len(X) * gini(left) + (len(right) / len(X)) * gini(right))
        gain = float(parent_gini - weighted_child)

        if gain > best_gain:
            best_feature, best_gain = j, gain

    return (best_feature, best_gain)

if __name__ == "__main__":
    X = np.array([[1,1],[1,0],[0,1],[0,0]])
    y = np.array([1,1,0,0])
    print(best_split(X, y))   # expect (0, 0.5)
