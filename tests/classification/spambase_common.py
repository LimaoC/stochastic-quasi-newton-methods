import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call, grad, vmap

from sqnm.utils.param import flatten, unflatten

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def linear_model(d):
    model = nn.Linear(d, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.zeros(d, dtype=torch.float32))
    return model


def load_spambase_data(rng: np.random.Generator):
    file = Path(__file__).parent / "data/spambase/spambase.data"
    with open(file, "rb") as file_handle:
        raw_data = np.loadtxt(file_handle, delimiter=",", skiprows=0)

    raw_train = raw_data[:, :-1]
    raw_labels = raw_data[:, -1].astype(int)

    n = raw_labels.size
    test_size = np.ceil(n / 5).astype(int)
    test_index = rng.choice(n, test_size, replace=False)
    train_index = np.setdiff1d(np.arange(n), test_index)

    X_train = torch.tensor(raw_train[train_index, :], dtype=torch.float32)
    X_test = torch.tensor(raw_train[test_index, :], dtype=torch.float32)
    y_train = torch.tensor(raw_labels[train_index], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(raw_labels[test_index], dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test


def log_test_set_info(X_test, y_test, model, loss_fn, epoch, loss):
    params = {name: param.detach() for name, param in model.named_parameters()}
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    prob = torch.sigmoid(functional_call(model, params, (X_test,)))
    pred = prob > 0.5
    acc = (pred == y_test).float().mean().item()

    def compute_loss(inputs: Tensor) -> Tensor:
        pred = functional_call(model, unflatten(inputs, param_shapes), (X_test,))
        return loss_fn(pred, y_test)

    grad_fn = torch.func.grad(compute_loss)
    grad_norm = grad_fn(flatten(params)).norm()

    logger.info(
        f"Epoch {epoch:4}, loss = {loss:.4f}, acc = {acc:.4f}, "
        f"grad norm = {grad_norm:.4f}"
    )


def create_ft_closure(
    optimizer, model, loss_fn, per_sample_loss_fn, params, buffers, X_batch, y_batch
):

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        target = target.unsqueeze(0)

        pred = functional_call(model, (params, buffers), (batch,))
        loss = loss_fn(pred, target)
        return loss

    def closure(estimate_var: bool = False) -> float:
        optimizer.zero_grad()
        if not estimate_var:
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            return loss.item()

        per_sample_loss = per_sample_loss_fn(model(X_batch), y_batch)
        loss = per_sample_loss.mean()
        loss.backward()

        per_sample_grad_fn = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))
        per_sample_grad = per_sample_grad_fn(params, buffers, X_batch, y_batch)
        flat_per_sample_grad = [
            g.reshape(g.shape[0], -1) for g in per_sample_grad.values()
        ]
        grad_mat = torch.cat(flat_per_sample_grad, dim=1)

        loss_var = per_sample_loss.var(unbiased=True)
        grad_var = grad_mat.var(dim=0, unbiased=True)

        return loss.item(), grad_mat.mean(dim=0), loss_var, grad_var

    return closure
