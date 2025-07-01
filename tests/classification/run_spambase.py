import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from sqnm.optim.lbfgs import LBFGS
from sqnm.optim.olbfgs import OLBFGS
from sqnm.optim.sqn_hv import SQNHv

MAX_EPOCHS = 1_000

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def load_data(file: str, rng: np.random.Generator):
    with open(file, "rb") as file_handle:
        raw_data = np.loadtxt(file_handle, delimiter=",", skiprows=0)

    raw_train = raw_data[:, :-1]
    raw_labels = raw_data[:, -1].astype(int)

    n = raw_labels.size
    test_size = np.ceil(n / 5).astype(int)
    test_index = rng.choice(n, test_size, replace=False)
    train_index = np.setdiff1d(np.arange(n), test_index)

    A_train = torch.tensor(raw_train[train_index, :], dtype=torch.float32)
    A_test = torch.tensor(raw_train[test_index, :], dtype=torch.float32)
    b_train = torch.tensor(raw_labels[train_index], dtype=torch.float32)
    b_test = torch.tensor(raw_labels[test_index], dtype=torch.float32)

    return A_train, A_test, b_train, b_test


def log_test_set_info(A_test, b_test, x, epoch, loss):
    with torch.no_grad():
        logits = A_test @ x
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        acc = (preds == b_test).float().mean().item()
    grad_norm = x.grad.norm()

    logger.info(
        f"Epoch {epoch+1:4}, loss = {loss:.4f}, acc = {acc:.4f}, "
        f"grad norm = {grad_norm:.4f}"
    )


def run_sgd(A_train, A_test, b_train, b_test, batch_size, lr) -> list[float]:
    n, d = A_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    x = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = SGD([x], lr=lr)

    train_dataset = TensorDataset(A_train, b_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for A_batch, b_batch in train_dataloader:
            optimizer.zero_grad()
            logits = A_batch @ x
            loss = loss_fn(logits, b_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        losses.append(epoch_loss)

        if epoch % 100 == 99:
            log_test_set_info(A_test, b_test, x, epoch, epoch_loss)

    return losses


def run_lbfgs(A_train, A_test, b_train, b_test) -> list[float]:
    n, d = A_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    x = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = LBFGS([x], line_search_fn="strong_wolfe")

    def closure() -> float:
        optimizer.zero_grad()
        logits = A_train @ x
        loss = loss_fn(logits, b_train)
        loss.backward()
        return loss.item()

    losses = []
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        losses.append(loss)

        if epoch % 100 == 99:
            log_test_set_info(A_test, b_test, x, epoch, loss)

    return losses


def run_olbfgs(A_train, A_test, b_train, b_test, batch_size) -> list[float]:
    n, d = A_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    x = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = OLBFGS(
        [x],
        history_size=4,
        eps=1e-10,
        eta0=0.1 * batch_size / (batch_size + 2),
        tau=10,
        c=1,
    )

    train_dataset = TensorDataset(A_train, b_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for A_batch, b_batch in train_dataloader:

            def closure() -> float:
                optimizer.zero_grad()
                logits = A_batch @ x
                loss = loss_fn(logits, b_batch)
                loss.backward()
                return loss.item()

            optimizer.zero_grad()
            loss = optimizer.step(closure)

            epoch_loss += loss
            num_batches += 1

        losses.append(epoch_loss / num_batches)

        if epoch % 100 == 99:
            log_test_set_info(A_test, b_test, x, epoch, loss)

    return losses


def run_sqn_hv(
    A_train, A_test, b_train, b_test, batch_size, curvature_batch_size, skip
) -> list[float]:
    n, d = A_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    x = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = SQNHv([x], history_size=5, skip=skip)

    train_dataset = TensorDataset(A_train, b_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    k = 1
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for A_batch, b_batch in train_dataloader:

            def closure() -> Tensor:
                optimizer.zero_grad()
                logits = A_batch @ x
                loss = loss_fn(logits, b_batch)
                loss.backward()
                return loss.item()

            if k % skip == 0:
                idx = torch.randint(0, n, (curvature_batch_size,))
                A_batch_curvature, b_batch_curvature = A_train[idx], b_train[idx]

                def curvature_f(x: Tensor) -> Tensor:
                    logits = A_batch_curvature @ x
                    loss = loss_fn(logits, b_batch_curvature)
                    return loss

                optimizer.zero_grad()
                loss = optimizer.step(closure, curvature_f)
            else:
                optimizer.zero_grad()
                loss = optimizer.step(closure)

            # if k > 2 * skip:
            #     print("sqnhv", loss)

            epoch_loss += loss
            num_batches += 1
            k += 1

        losses.append(epoch_loss / num_batches)

        if epoch % 100 == 99:
            log_test_set_info(A_test, b_test, x, epoch, loss)

    return losses


def main():
    rng = np.random.default_rng(17)
    torch.manual_seed(17)

    data_file = Path(__file__).parent / "data/spambase/spambase.data"
    A_train, A_test, b_train, b_test = load_data(data_file, rng)

    logger.info(f"A_train = {A_train.shape}")
    logger.info(f"b_train = {b_train.shape}")
    logger.info(f"A_test  = {A_test.shape}")
    logger.info(f"b_test  = {b_test.shape}")

    methods = [
        ("SGD", run_sgd, dict(batch_size=50, lr=1e-4)),
        ("L-BFGS", run_lbfgs, dict()),
        ("oL-BFGS", run_olbfgs, dict(batch_size=200)),
        ("SQN-Hv", run_sqn_hv, dict(batch_size=100, curvature_batch_size=600, skip=10)),
    ]
    method_losses = []

    for method in methods:
        start = time.time()
        name, func, params = method
        losses = func(A_train, A_test, b_train, b_test, **params)
        logger.info(f"{name} took {time.time() - start:.3f} seconds")
        method_losses.append(losses)

    for i, method in enumerate(methods):
        plt.plot(range(len(method_losses[i])), method_losses[i], label=method[0])
    plt.ylim((0, 1.1))
    plt.title("Losses vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
