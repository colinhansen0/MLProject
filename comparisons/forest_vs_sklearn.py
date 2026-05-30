"""Validate the from-scratch RandomForest against sklearn on the same BACE features."""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from core.data import get_dataset
from core.RF.forest import RandomForest


def main() -> None:
    X, y_class, _ = get_dataset()
    y = y_class.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y,
    )

    ours = RandomForest(n_estimators=100, random_state=0).fit(X_train, y_train)
    ours_pred = ours.predict(X_test)

    skl = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
    skl_pred = skl.predict(X_test)

    print(f"ours    accuracy: {accuracy_score(y_test, ours_pred):.4f}")
    print(f"sklearn accuracy: {accuracy_score(y_test, skl_pred):.4f}")
    print(f"agreement (ours vs sklearn): {(ours_pred == skl_pred).mean():.4f}")


if __name__ == "__main__":
    main()