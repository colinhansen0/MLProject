"""Validate the from-scratch regression tree and forest against sklearn on BACE pIC50."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from core.data import get_dataset
from core.RF.tree import DecisionTree, REGRESSION
from core.RF.forest import RandomForest


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    X, _, y_reg = get_dataset()          # continuous pIC50 target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=0,   # no stratify — continuous target
    )

    # single tree
    ours_tree = DecisionTree(task=REGRESSION).fit(X_train, y_train)
    skl_tree = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
    ours_tree_pred = ours_tree.predict(X_test)
    skl_tree_pred = skl_tree.predict(X_test)

    # forest
    ours_forest = RandomForest(n_estimators=100, random_state=0, task=REGRESSION).fit(X_train, y_train)
    skl_forest = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
    ours_forest_pred = ours_forest.predict(X_test)
    skl_forest_pred = skl_forest.predict(X_test)

    print(f"tree   ours    R2: {r2_score(y_test, ours_tree_pred):.4f}   RMSE: {rmse(y_test, ours_tree_pred):.4f}")
    print(f"tree   sklearn R2: {r2_score(y_test, skl_tree_pred):.4f}   RMSE: {rmse(y_test, skl_tree_pred):.4f}")
    print(f"forest ours    R2: {r2_score(y_test, ours_forest_pred):.4f}   RMSE: {rmse(y_test, ours_forest_pred):.4f}")
    print(f"forest sklearn R2: {r2_score(y_test, skl_forest_pred):.4f}   RMSE: {rmse(y_test, skl_forest_pred):.4f}")


if __name__ == "__main__":
    main()