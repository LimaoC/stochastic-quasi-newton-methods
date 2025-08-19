import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.func as ft
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.sgd import SGD
from torch.utils.data import TensorDataset

from sqnm.optim.mbbfgs import MBBFGS
from sqnm.optim.olbfgs import OLBFGS
from sqnm.optim.sqn_hv import SQNHv

from ..train_mbbfgs import train as train_mbbfgs
from ..train_olbfgs import train as train_olbfgs
from ..train_sgd import train as train_sgd
from ..train_sqn_hv import train as train_sqn_hv
from ..train_util import get_device, timing_context

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
    parser = argparse.ArgumentParser(prog="run_spambase")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument(
        "-s",
        "--step_size",
        choices=["decaying", "strong_wolfe", "prob_wolfe"],
        default="decaying",
    )
    parser.add_argument("--save-fig", action="store_true")
    args = parser.parse_args()

    num_epochs = args.epochs
    log_frequency = max(min(num_epochs // 10, 100), 1)
    batch_size = args.batch_size
    step_size_strategy = args.step_size

    device = "cpu"  # get_device()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    gen = torch.Generator(device)

    X_train, X_test, y_train, y_test = load_spambase_data(rng)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    logger.info(f"X_train = {X_train.shape}")
    logger.info(f"y_train = {y_train.shape}")
    logger.info(f"X_test  = {X_test.shape}")
    logger.info(f"y_test  = {y_test.shape}")

    n, d = X_train.shape
    loss_fn = nn.BCEWithLogitsLoss()

    ####################################################################################

    with timing_context("SGD"):
        model = linear_model(d).to(device)
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
        if step_size_strategy == "decaying":
            scheduler = ExponentialLR(optimizer, 0.99)
        else:
            scheduler = None
        sgd_out = train_sgd(
            train_dataset,
            test_dataset,
            optimizer,
            scheduler,
            model,
            loss_fn,
            device,
            generator=gen,
            num_epochs=num_epochs,
            log_frequency=log_frequency,
            batch_size=batch_size,
        )

    with timing_context("oL-BFGS"):
        model = linear_model(d).to(device)
        if step_size_strategy == "decaying":
            optimizer = OLBFGS(model.parameters(), lr=1e-1, reg_term=1.0)
            scheduler = ExponentialLR(optimizer, 0.99)
        elif step_size_strategy == "strong_wolfe":
            optimizer = OLBFGS(
                model.parameters(),
                lr=1e-1,
                line_search_fn=step_size_strategy,
                reg_term=1.0,
            )
            scheduler = None
        olbfgs_out = train_olbfgs(
            train_dataset,
            test_dataset,
            optimizer,
            scheduler,
            model,
            loss_fn,
            device,
            generator=gen,
            num_epochs=num_epochs,
            log_frequency=log_frequency,
            batch_size=batch_size,
        )

    with timing_context("SQN-Hv"):
        model = linear_model(d).to(device)
        if step_size_strategy == "decaying":
            optimizer = SQNHv(model.parameters(), lr=1e-3)
            scheduler = ExponentialLR(optimizer, 0.99)
        elif step_size_strategy == "strong_wolfe":
            optimizer = SQNHv(
                model.parameters(), lr=1e-3, line_search_fn=step_size_strategy
            )
            scheduler = None
        sqnhv_out = train_sqn_hv(
            train_dataset,
            test_dataset,
            optimizer,
            scheduler,
            model,
            loss_fn,
            device,
            generator=gen,
            num_epochs=num_epochs,
            log_frequency=log_frequency,
            batch_size=batch_size,
            curvature_batch_size=batch_size * 2,
        )

    with timing_context("MB-BFGS"):
        model = linear_model(d).to(device)
        if step_size_strategy == "decaying":
            optimizer = MBBFGS(model.parameters(), lr=1e-3)
            scheduler = ExponentialLR(optimizer, 0.99)
        mbbfgs_out = train_mbbfgs(
            train_dataset,
            test_dataset,
            optimizer,
            scheduler,
            model,
            loss_fn,
            device,
            generator=gen,
            num_epochs=num_epochs,
            log_frequency=log_frequency,
            batch_size=batch_size,
            overlap_batch_size=batch_size // 5,
        )

    ####################################################################################

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    skip = 10
    epochs = np.arange(skip, num_epochs)
    (p1,) = ax.plot(epochs, sgd_out["epoch_losses"][skip:], label="SGD")
    (p2,) = ax.plot(epochs, olbfgs_out["epoch_losses"][skip:], label="oL-BFGS")
    (p3,) = ax.plot(epochs, sqnhv_out["epoch_losses"][skip:], label="SQN-Hv")
    (p4,) = ax.plot(epochs, mbbfgs_out["epoch_losses"][skip:], label="MB-BFGS")

    epochs = np.arange(0, num_epochs, log_frequency) + log_frequency
    colors = [p1.get_color(), p2.get_color(), p3.get_color(), p4.get_color()]
    ax.plot(epochs, sgd_out["test_losses"], color=colors[0], marker="x", alpha=0.3)
    ax.plot(epochs, olbfgs_out["test_losses"], color=colors[1], marker="x", alpha=0.3)
    ax.plot(epochs, sqnhv_out["test_losses"], color=colors[2], marker="x", alpha=0.3)
    ax.plot(epochs, mbbfgs_out["test_losses"], color=colors[3], marker="x", alpha=0.3)

    ax.set_title(
        f"Losses vs. epochs ({step_size_strategy}, {num_epochs} epochs, "
        f"{batch_size} batch size)"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # ax.set_xscale("log")
    ax.legend()

    if args.save_fig:
        plt.savefig(
            f"./figures/logistic_regression/"
            f"{step_size_strategy}-{num_epochs}epochs-{batch_size}-batch.pdf"
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
