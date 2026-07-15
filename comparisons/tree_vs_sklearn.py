"""Validate the from-scratch DecisionTree against sklearn on the same BACE features.

Both models train on one shared train split and score on one shared test split,
so any gap is the algorithm, not the data. Expect *comparable* accuracy, not
identical trees — greedy tie-breaking and feature ordering legitimately differ.
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from core.data import get_dataset
from core.tree import DecisionTree

def main() -> None:
    X, y_class, _ = get_dataset()          # binary target; ignore pIC50 for now
    y = y_class.astype(int)                # np.bincount needs integer labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,                        # keep the class balance in both splits
    )

    ours = DecisionTree().fit(X_train, y_train)
    ours_pred = ours.predict(X_test)

    skl = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X_train, y_train)
    skl_pred = skl.predict(X_test)

    ours_correct_count = float(0)
    sk_correct_count = float(0)
    total_count = len(y_test)

    for pred, truth in zip (ours_pred, y_test):
        if pred == truth:
            ours_correct_count += 1

    for pred, truth in zip (skl_pred, y_test):
        if pred == truth:
            sk_correct_count += 1

    print(f"ours    accuracy: {accuracy_score(y_test, ours_pred):.4f}")
    print(f"sklearn accuracy: {accuracy_score(y_test, skl_pred):.4f}")
    print(f"agreement (ours vs sklearn): {(ours_pred == skl_pred).mean():.4f}")

if __name__ == "__main__":
    main()