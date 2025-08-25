import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.sgd import SGD
from torch.utils.data import TensorDataset
from torcheval.metrics import BinaryAccuracy

from sqnm.optim.mbbfgs import MBBFGS
from sqnm.optim.olbfgs import OLBFGS
from sqnm.optim.scbfgs import SCBFGS
from sqnm.optim.sqnhv import SQNHv

from ..train_mbbfgs import train as train_mbbfgs
from ..train_olbfgs import train as train_olbfgs
from ..train_scbfgs import train as train_scbfgs
from ..train_sgd import train as train_sgd
from ..train_sqnhv import train as train_sqnhv
from ..train_util import (
    compute_loss,
    get_device,
    parameter_grid,
    save_lowest_test_loss_runs,
    timing_context,
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "./outs/binary_classification/spambase/objs/"
DIMS = 57


def linear_model() -> nn.Linear:
    model = nn.Linear(DIMS, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.zeros(DIMS, dtype=torch.float32))
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


def main():
    parser = argparse.ArgumentParser(prog="run_spambase")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=100, nargs="+")
    parser.add_argument(
        "-s",
        "--step_size",
        choices=["decaying", "strong_wolfe", "prob_wolfe"],
        default="decaying",
    )
    parser.add_argument(
        "-S",
        "--skip",
        choices=["sgd", "olbfgs", "sqnhv", "mbbfgs", "scbfgs"],
        default=[],
        nargs="+",
    )
    args = parser.parse_args()

    # Hyperparameters
    num_epochs = args.epochs
    batch_sizes = args.batch_size
    step_size = args.step_size
    log_frequency = max(min(num_epochs // 10, 100), 1)
    lrs = [1e-2, 1e-4]  # Learning rates for decaying step size
    lr_gamma = 0.99

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

    loss_fn = nn.BCEWithLogitsLoss()
    metrics = {"accuracy": BinaryAccuracy()}

    initial_loss = compute_loss(test_dataset, linear_model(), loss_fn)

    ####################################################################################

    def run_sgd(params):
        batch_size, lr = params["batch_size"], params["lr"]
        model = linear_model().to(device)
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        scheduler = ExponentialLR(optimizer, lr_gamma)
        with timing_context(f"SGD ({params=})"):
            out = train_sgd(
                train_dataset,
                test_dataset,
                optimizer,
                scheduler,
                model,
                loss_fn,
                device,
                metrics=metrics,
                generator=gen,
                num_epochs=num_epochs,
                log_frequency=log_frequency,
                batch_size=batch_size,
            )
        out["epoch_losses"].insert(0, initial_loss)
        return out

    def run_olbfgs(params):
        model = linear_model().to(device)
        batch_size = params["batch_size"]
        reg = params["reg_term"]
        c = params["c"]
        if step_size == "decaying":
            lr = params["lr"]
            optimizer = OLBFGS(model.parameters(), lr=lr, reg_term=reg, c=c)
            scheduler = ExponentialLR(optimizer, lr_gamma)
        elif step_size == "strong_wolfe":
            optimizer = OLBFGS(
                model.parameters(),
                line_search_fn=step_size,
                reg_term=reg,
                c=c,
            )
            scheduler = None
        with timing_context(f"oL-BFGS ({step_size}, {params=})"):
            out = train_olbfgs(
                train_dataset,
                test_dataset,
                optimizer,
                scheduler,
                model,
                loss_fn,
                device,
                metrics=metrics,
                generator=gen,
                num_epochs=num_epochs,
                log_frequency=log_frequency,
                batch_size=batch_size,
            )
        out["epoch_losses"].insert(0, initial_loss)
        return out

    def run_sqnhv(params):
        model = linear_model().to(device)
        batch_size, skip, curv_batch_size = params["batch_size|skip|curv_batch_size"]
        if step_size == "decaying":
            lr = params["lr"]
            optimizer = SQNHv(model.parameters(), lr=lr, skip=skip)
            scheduler = ExponentialLR(optimizer, lr_gamma)
        elif step_size == "strong_wolfe":
            optimizer = SQNHv(model.parameters(), line_search_fn=step_size, skip=skip)
            scheduler = None
        with timing_context(f"SQN-Hv ({step_size}, {params=})"):
            out = train_sqnhv(
                train_dataset,
                test_dataset,
                optimizer,
                scheduler,
                model,
                loss_fn,
                device,
                metrics=metrics,
                generator=gen,
                num_epochs=num_epochs,
                log_frequency=log_frequency,
                batch_size=batch_size,
                curvature_batch_size=curv_batch_size,
            )
        out["epoch_losses"].insert(0, initial_loss)
        return out

    def run_mbbfgs(params):
        model = linear_model().to(device)
        batch_size, overlap = params["batch_size"], params["overlap_percent"]
        if step_size == "decaying":
            lr = params["lr"]
            optimizer = MBBFGS(model.parameters(), lr=lr)
            scheduler = ExponentialLR(optimizer, lr_gamma)
        elif step_size == "strong_wolfe":
            optimizer = MBBFGS(model.parameters(), line_search_fn=step_size)
            scheduler = None
        with timing_context(f"MB-BFGS ({step_size}, {params=})"):
            out = train_mbbfgs(
                train_dataset,
                test_dataset,
                optimizer,
                scheduler,
                model,
                loss_fn,
                device,
                metrics=metrics,
                generator=gen,
                num_epochs=num_epochs,
                log_frequency=log_frequency,
                batch_size=batch_size,
                overlap_batch_size=int(batch_size * overlap),
            )
        out["epoch_losses"].insert(0, initial_loss)
        return out

    def run_scbfgs(params):
        model = linear_model().to(device)
        batch_size = params["batch_size"]
        eta1 = params["eta1"]
        rho = params["rho"]
        tau = params["tau"]
        if step_size == "decaying":
            lr = params["lr"]
            optimizer = SCBFGS(model.parameters(), lr=lr, eta1=eta1, rho=rho, tau=tau)
            scheduler = ExponentialLR(optimizer, lr_gamma)
        elif step_size == "strong_wolfe":
            optimizer = SCBFGS(
                model.parameters(),
                line_search_fn=step_size,
                eta1=eta1,
                rho=rho,
                tau=tau,
            )
            scheduler = None
        with timing_context(f"SC-BFGS ({step_size}, {params=})"):
            out = train_scbfgs(
                train_dataset,
                test_dataset,
                optimizer,
                scheduler,
                model,
                loss_fn,
                device,
                metrics=metrics,
                generator=gen,
                num_epochs=num_epochs,
                log_frequency=log_frequency,
                batch_size=batch_size,
            )
        out["epoch_losses"].insert(0, initial_loss)
        return out

    if "sgd" not in args.skip:
        outs = []
        param_dict = {"batch_size": batch_sizes, "lr": lrs}
        for params in parameter_grid(param_dict):
            out = run_sgd(params)
            # Discard this run if it ran for less than num_epochs // 2
            if len(out["epoch_losses"]) >= num_epochs // 2:
                out["params"] = params
                outs.append(out)
        save_lowest_test_loss_runs(outs, "sgd", step_size, OUTPUT_DIR)

    if "olbfgs" not in args.skip:
        outs = []
        param_dict = {
            "batch_size": batch_sizes,
            "reg_term": [0.0, 0.1, 1.0],
            "c": [0.1, 1.0],
        }
        if step_size == "decaying":
            param_dict["lr"] = lrs
        for params in parameter_grid(param_dict):
            out = run_olbfgs(params)
            # Discard this run if it ran for less than num_epochs // 2
            if len(out["epoch_losses"]) >= num_epochs // 2:
                out["params"] = params
                outs.append(out)
        save_lowest_test_loss_runs(outs, "olbfgs", step_size, OUTPUT_DIR)

    if "sqnhv" not in args.skip:
        outs = []
        batch_skip_curv = [
            (batch_size, skip, skip * batch_size // curv_factor)
            for skip in [10, 20]  # skip
            for batch_size in batch_sizes
            for curv_factor in [2, 10, 20]  # curvature batch size scale
        ]
        param_dict = {"batch_size|skip|curv_batch_size": batch_skip_curv}
        if step_size == "decaying":
            param_dict["lr"] = lrs
        for params in parameter_grid(param_dict):
            out = run_sqnhv(params)
            # Discard this run if it ran for less than num_epochs // 2
            if len(out["epoch_losses"]) >= num_epochs // 2:
                out["params"] = params
                outs.append(out)
        save_lowest_test_loss_runs(outs, "sqnhv", step_size, OUTPUT_DIR)

    if "mbbfgs" not in args.skip:
        outs = []
        param_dict = {"batch_size": batch_sizes, "overlap_percent": [0.05, 0.1, 0.3]}
        if step_size == "decaying":
            param_dict["lr"] = lrs
        for params in parameter_grid(param_dict):
            out = run_mbbfgs(params)
            # Discard this run if it ran for less than num_epochs // 2
            if len(out["epoch_losses"]) >= num_epochs // 2:
                out["params"] = params
                outs.append(out)
        save_lowest_test_loss_runs(outs, "mbbfgs", step_size, OUTPUT_DIR)

    if "scbfgs" not in args.skip:
        outs = []
        param_dict = {
            "batch_size": batch_sizes,
            "eta1": [1 / 4, 1 / 64],
            "rho": [1 / 8, 1 / 128],
            "tau": [8, 16],
        }
        if step_size == "decaying":
            param_dict["lr"] = lrs
        for params in parameter_grid(param_dict):
            out = run_scbfgs(params)
            # Discard this run if it ran for less than num_epochs // 2
            if len(out["epoch_losses"]) >= num_epochs // 2:
                out["params"] = params
                outs.append(out)
        save_lowest_test_loss_runs(outs, "scbfgs", step_size, OUTPUT_DIR)


if __name__ == "__main__":
    main()
