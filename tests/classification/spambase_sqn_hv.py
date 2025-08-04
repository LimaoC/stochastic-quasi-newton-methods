import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.func as ft
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from sqnm.optim.sqn_hv import SQNHv
from sqnm.utils.param import unflatten

from .spambase_common import linear_model, load_spambase_data, log_test_set_info

MAX_EPOCHS = 1000
BATCH_SIZE = 100
CURVATURE_BATCH_SIZE = 600
SKIP = 10

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def run(X_train, X_test, y_train, y_test, rng) -> list[float]:
    n, d = X_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    model = linear_model(d)
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    optimizer = SQNHv(model.parameters(), history_size=20, skip=SKIP, beta=1e-1)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    k = 1
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_dataloader:

            def closure() -> float:
                """Re-evaluates and returns model loss, populates gradients"""
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch), y_batch)
                loss.backward()
                return loss.item()

            if k % SKIP == 0:
                idx = rng.choice(n, CURVATURE_BATCH_SIZE, replace=False)
                X_curv_batch, y_curv_batch = X_train[idx], y_train[idx]

                def curvature_fn(inputs: Tensor) -> Tensor:
                    """Pure function to return loss."""
                    inputs = unflatten(inputs, param_shapes)
                    pred = ft.functional_call(model, inputs, (X_curv_batch,))
                    loss = loss_fn(pred, y_curv_batch)
                    return loss

                loss = optimizer.step(closure, curvature_fn=curvature_fn)
            else:
                loss = optimizer.step(closure)

            epoch_loss += loss
            num_batches += 1
            k += 1

        losses.append(epoch_loss / num_batches)

        if epoch % 100 == 99:
            log_test_set_info(X_test, y_test, model, loss_fn, epoch + 1, loss)

    return losses


def run_ls(X_train, X_test, y_train, y_test, rng) -> list[float]:
    n, d = X_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    ps_loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # per-sample
    model = linear_model(d)
    params = {name: param.detach() for name, param in model.named_parameters()}
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    optimizer = SQNHv(
        model.parameters(), history_size=20, line_search_fn="prob_wolfe", skip=SKIP
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def compute_loss_one_sample(params, inp, target):
        """Pure function to compute loss for a single sample"""
        # Model expects batch dimension
        inp_batch = inp.unsqueeze(0)
        target_batch = target.unsqueeze(0)

        pred = ft.functional_call(model, params, (inp_batch,))
        loss = loss_fn(pred, target_batch)
        return loss

    losses = []
    k = 1
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_dataloader:

            def closure() -> float:
                """Re-evaluates and returns model loss, populates gradients"""
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch), y_batch)
                loss.backward()
                return loss.item()

            def fn(
                inputs: Tensor, return_vars: bool
            ) -> Tensor | tuple[float, Tensor, float, Tensor]:
                """
                Pure function to return loss. If return_vars is True, also returns
                gradient, loss var and gradient var
                """
                inputs = unflatten(inputs, param_shapes)
                pred = ft.functional_call(model, inputs, (X_batch,))

                if not return_vars:
                    loss = loss_fn(pred, y_batch)
                    return loss

                ps_loss = ps_loss_fn(pred, y_batch)  # [b, 1]
                loss = ps_loss.mean().item()

                ps_grad_fn = ft.vmap(
                    ft.grad(compute_loss_one_sample), in_dims=(None, 0, 0)
                )
                ps_grad = ps_grad_fn(params, X_batch, y_batch)
                flat_ps_grad = [g.reshape(g.shape[0], -1) for g in ps_grad.values()]
                grad_matrix = torch.cat(flat_ps_grad, dim=1)  # [b, d]
                grad_mean = grad_matrix.mean(dim=0)  # [d]

                loss_var = ps_loss.var(unbiased=True).item()
                grad_var = grad_matrix.var(dim=0, unbiased=True)

                return loss, grad_mean, loss_var, grad_var

            if k % SKIP == 0:
                idx = rng.choice(n, CURVATURE_BATCH_SIZE, replace=False)
                X_curv_batch, y_curv_batch = X_train[idx], y_train[idx]

                def curvature_fn(inputs: Tensor) -> Tensor:
                    """Pure function to return loss."""
                    inputs = unflatten(inputs, param_shapes)
                    pred = ft.functional_call(model, inputs, (X_curv_batch,))
                    loss = loss_fn(pred, y_curv_batch)
                    return loss

                loss = optimizer.step(closure, fn, curvature_fn)
            else:
                loss = optimizer.step(closure, fn)

            epoch_loss += loss
            num_batches += 1
            k += 1

        losses.append(epoch_loss / num_batches)

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
    logger.info("Training with SQN-Hv...")
    losses = run_ls(X_train, X_test, y_train, y_test, rng)
    logger.info(f"SQN-Hv took {time.time() - start:.3f} seconds")

    plt.plot(range(len(losses)), losses)
    plt.ylim((0, 1.1))
    plt.title("Losses vs. epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
