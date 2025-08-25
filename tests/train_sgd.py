import logging
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.metric import Metric

from .train_util import compute_loss, compute_metric, create_closure, log_training_info

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: SGD,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    metrics: dict[str, Metric] = dict(),
    generator: torch.Generator | None = None,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
) -> dict[str, Any]:
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    num_batches = len(dataloader)

    epoch_losses = []
    test_losses = []  # computed every log_frequency epochs
    test_metrics = defaultdict(list)  # problem-specific metrics
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
            epoch_loss += optimizer.step(closure)

        if scheduler is not None:
            scheduler.step()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)

        if math.isnan(epoch_loss):
            logger.info(f"\tLoss hit NaN at epoch {epoch+1}, stopping early")
            break

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn)
            test_losses.append(test_loss)
            for name, metric in metrics.items():
                metric_val = compute_metric(test_dataset, model, metric)
                test_metrics[name].append(metric_val)
            log_training_info(epoch + 1, epoch_loss, test_loss, test_metrics)

    out = {
        "epoch_losses": epoch_losses,
        "test_losses": test_losses,
        "min_test_loss": float("inf") if not test_losses else min(test_losses),
        "log_frequency": log_frequency,
    }
    for name, metric in test_metrics.items():
        out[name] = test_metrics[name]
    return out
