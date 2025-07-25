import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.func as ft
import torch.nn as nn
from torch import Tensor

from sqnm.optim.lbfgs import LBFGS
from sqnm.utils.param import unflatten

from .spambase_common import linear_model, load_spambase_data, log_test_set_info

MAX_EPOCHS = 1000

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def run(X_train, X_test, y_train, y_test) -> list[float]:
    n, d = X_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    model = linear_model(d)
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    optimizer = LBFGS(model.parameters(), line_search_fn="strong_wolfe")

    def closure() -> float:
        """Re-evaluates and returns model loss, populates gradients"""
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        return loss.item()

    def fn(inputs: Tensor) -> Tensor:
        """Pure function to compute loss"""
        pred = ft.functional_call(model, unflatten(inputs, param_shapes), (X_train,))
        return loss_fn(pred, y_train)

    losses = []
    for epoch in range(MAX_EPOCHS):
        loss = optimizer.step(closure, fn)
        losses.append(loss)

        if epoch % 100 == 99:
            log_test_set_info(X_test, y_test, model, loss_fn, epoch + 1, loss)

    return losses


def main():
    rng = np.random.default_rng(17)
    torch.manual_seed(17)

    X_train, X_test, y_train, y_test = load_spambase_data(rng)

    logger.info(f"X_train = {X_train.shape}")
    logger.info(f"y_train = {y_train.shape}")
    logger.info(f"X_test  = {X_test.shape}")
    logger.info(f"y_test  = {y_test.shape}")

    start = time.time()
    logger.info("Training with L-BFGS...")
    losses = run(X_train, X_test, y_train, y_test)
    logger.info(f"L-BFGS took {time.time() - start:.3f} seconds")

    plt.plot(range(len(losses)), losses)
    plt.ylim((0, 1.1))
    plt.title("Losses vs. epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
