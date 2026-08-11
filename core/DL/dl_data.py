from __future__ import annotations

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[0] / "data" / "mnist.csv"


def load_and_split(
    path: Path = DATA_PATH, rebuild: bool = False) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """load data from path specified, rebuild if rebuild = True or path doesnt exist
    traing val test split 71:14:14 or 50k:10k:10k
    """

    if rebuild or not path.exists():
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.column_stack([y.astype(np.uint8), X.astype(np.uint8)])
        np.savetxt(path, raw, delimiter=",", fmt="%d")

    data = pd.read_csv(path, header=None, dtype=np.uint8).to_numpy()

    y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0

    X_train, y_train = X[:50000], y[:50000]
    X_val,   y_val   = X[50000:60000], y[50000:60000]
    X_test,  y_test  = X[60000:], y[60000:]

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split()

    print("train:", X_train.shape, y_train.shape)
    print("val:  ", X_val.shape, y_val.shape)
    print("test: ", X_test.shape, y_test.shape)
    print("pixel range:  ", X_train.min(), X_train.max())     # expect 0.0 1.0
    print("labels:       ", y_train[:10])                     # expect 5 0 4 1 9 2 1 3 1 4
    print("label range:  ", y_train.min(), y_train.max())     # expect 0 9
    print("class balance:", np.bincount(y_train))             # ~5000 each