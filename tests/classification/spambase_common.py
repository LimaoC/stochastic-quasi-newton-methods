import logging
from pathlib import Path

import numpy as np
import torch
import torch.func as ft
import torch.nn as nn
from torch import Tensor

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

    prob = torch.sigmoid(ft.functional_call(model, params, (X_test,)))
    pred = prob > 0.5
    acc = (pred == y_test).float().mean().item()

    def compute_loss(inputs: Tensor) -> Tensor:
        pred = ft.functional_call(model, unflatten(inputs, param_shapes), (X_test,))
        return loss_fn(pred, y_test)

    grad_fn = ft.grad(compute_loss)
    grad_norm = grad_fn(flatten(params)).norm()

    logger.info(
        f"Epoch {epoch:4}, loss = {loss:.4f}, acc = {acc:.4f}, "
        f"grad norm = {grad_norm:.4f}"
    )
