import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dl_data import load_and_split


def main() -> None:
    (X_train, y_train), _, (X_test, y_test) = load_and_split()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(np.eye(10)[y_train], dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(784, 30), nn.Sigmoid(),
        nn.Linear(30, 10), nn.Sigmoid(),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=3.0)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=True)

    for epoch in range(30):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = 0.5 * ((out - yb) ** 2).sum() / xb.shape[0]
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = (model(X_test).argmax(dim=1) == y_test).sum().item()
        print(f"Epoch {epoch}: {correct} / {len(y_test)}")

    train_correct = (model(X_train).argmax(dim=1) == y_train).sum()

    print(f"{train_correct} / 50000")


if __name__ == "__main__":
    main()