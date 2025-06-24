import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, TensorDataset

from sqnm.optim.lbfgs import LBFGS

BATCH_SIZE = 50
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


def log_training_info(A_test, b_test, x, epoch, loss):
    with torch.no_grad():
        logits = A_test @ x
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        acc = (preds == b_test).float().mean().item()

    grad_norm = x.grad.norm()
    logger.info(
        f"Epoch {epoch+1:4}, loss = {loss:.4f}, acc = {acc:.4f}, "
        f"grad norm = {grad_norm:10.4f}"
    )


def run_sgd(A_train, A_test, b_train, b_test) -> list[float]:
    n, d = A_train.shape
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = SGD([x], lr=1e-4)

    train_dataset = TensorDataset(A_train, b_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

        losses.append(epoch_loss / num_batches)

        if epoch % 100 == 99:
            log_training_info(A_test, b_test, x, epoch, loss)

    return losses


def run_lbfgs(A_train, A_test, b_train, b_test) -> list[float]:
    n, d = A_train.shape
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
    optimizer = LBFGS([x], line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = A_train @ x
        loss = loss_fn(logits, b_train)
        loss.backward()
        return loss.item()

    losses = []
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = closure()
        optimizer.step(closure)
        losses.append(loss)

        if epoch % 100 == 99:
            log_training_info(A_test, b_test, x, epoch, loss)

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

    start = time.time()
    sgd_losses = run_sgd(A_train, A_test, b_train, b_test)
    logger.info(f"SGD took {time.time() - start:.3f} seconds")
    start = time.time()
    lbfgs_losses = run_lbfgs(A_train, A_test, b_train, b_test)
    logger.info(f"LBFGS took {time.time() - start:.3f} seconds")

    plt.title("Losses vs. epochs")
    plt.plot(range(len(sgd_losses)), sgd_losses, label="SGD")
    plt.plot(range(len(lbfgs_losses)), lbfgs_losses, label="L-BFGS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
