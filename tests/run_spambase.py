import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.func as ft
import torch.nn as nn
from torch.optim.sgd import SGD

from sqnm.optim.lbfgs import LBFGS
from sqnm.optim.olbfgs import OLBFGS
from sqnm.optim.sqn_hv import SQNHv

from .train_lbfgs import train as train_lbfgs
from .train_olbfgs import train as train_olbfgs
from .train_sgd import train as train_sgd
from .train_sqn_hv import train as train_sqn_hv
from .train_util import get_device, timing_context

# from .train_olbfgs import train_with_prob_ls as train_olbfgs_with_prob_ls

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


def compute_acc(X, y, model) -> float:
    params = {name: param.detach() for name, param in model.named_parameters()}
    output = ft.functional_call(model, params, (X,))
    prob = torch.sigmoid(output)
    pred = prob > 0.5
    acc = (pred == y).float().mean().item()
    return acc


def main():
    num_epochs = 1000
    batch_size = 500

    device = get_device()
    rng = np.random.default_rng(17)
    torch.manual_seed(17)

    X_train, X_test, y_train, y_test = load_spambase_data(rng)
    logger.info(f"X_train = {X_train.shape}")
    logger.info(f"y_train = {y_train.shape}")
    logger.info(f"X_test  = {X_test.shape}")
    logger.info(f"y_test  = {y_test.shape}")

    n, d = X_train.shape
    loss_fn = nn.BCEWithLogitsLoss()
    # ps_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    ####################################################################################

    with timing_context("SGD"):
        model = linear_model(d).to(device)
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        sgd_results = train_sgd(
            X_train,
            y_train,
            optimizer,
            model,
            loss_fn,
            device,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

    with timing_context("L-BFGS"):
        model = linear_model(d).to(device)
        optimizer = LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        lbfgs_results = train_lbfgs(
            X_train,
            y_train,
            optimizer,
            model,
            loss_fn,
            device,
            num_epochs=num_epochs,
        )

    with timing_context("oL-BFGS"):
        model = linear_model(d).to(device)
        optimizer = OLBFGS(
            model.parameters(),
            reg_term=1.0,
            alpha0=0.1 * batch_size / (batch_size + 2),
        )
        olbfgs_results = train_olbfgs(
            X_train,
            y_train,
            optimizer,
            model,
            loss_fn,
            device,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # optimizer = OLBFGS(
        #     model.parameters(),
        #     line_search_fn="prob_wolfe",
        #     reg_term=1.0,
        # )
        # olbfgs_results = train_olbfgs_with_prob_ls(
        #     X_train,
        #     y_train,
        #     optimizer,
        #     model,
        #     loss_fn,
        #     ps_loss_fn,
        #     device,
        #     num_epochs=num_epochs,
        #     batch_size=batch_size,
        # )

    with timing_context("SQN-Hv"):
        model = linear_model(d).to(device)
        optimizer = SQNHv(model.parameters(), beta=1e-1)
        sqn_hv_results = train_sqn_hv(
            X_train,
            y_train,
            optimizer,
            model,
            loss_fn,
            device,
            rng,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

    ####################################################################################

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    skip = 5
    ax1.plot(sgd_results[0][skip:], label="SGD")
    ax1.plot(lbfgs_results[0][skip:], label="L-BFGS")
    ax1.plot(olbfgs_results[0][skip:], label="oL-BFGS")
    ax1.plot(sqn_hv_results[0], label="SQN-Hv")

    ax2.plot(sgd_results[1][skip:], label="SGD")
    ax2.plot(lbfgs_results[1][skip:], label="L-BFGS")
    ax2.plot(olbfgs_results[1][skip:], label="oL-BFGS")
    ax2.plot(sqn_hv_results[1], label="SQN-Hv")

    ax1.set_title("Losses vs. epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.set_ylim((0 - 0.2, 5.0 + 0.2))

    ax2.set_title("Grad norm vs. epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Grad norm")
    ax2.set_xscale("log")
    ax2.legend()

    plt.show()
    # plt.savefig("./spambase.pdf")


if __name__ == "__main__":
    main()
