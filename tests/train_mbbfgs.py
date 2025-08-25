import logging
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.metric import Metric

from sqnm.optim.mbbfgs import MBBFGS

from .train_util import (
    OverlappingBatchSampler,
    compute_loss,
    compute_metric,
    create_closure,
    create_loss_fn_pure,
    log_training_info,
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: MBBFGS,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    metrics: dict[str, Metric] = dict(),
    generator: torch.Generator | None = None,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
    overlap_batch_size=20,
) -> dict[str, Any]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    line_search_fn = optimizer.param_groups[0]["line_search_fn"]

    # Sample batches s.t. there is overlap between consecutive batches
    dataloader = DataLoader(
        train_dataset,
        batch_sampler=OverlappingBatchSampler(
            train_dataset, batch_size, overlap_batch_size, generator=generator
        ),
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

            # Overlap with next batch is at the end of this batch
            X_o = X_batch[-overlap_batch_size:]
            y_o = y_batch[-overlap_batch_size:]
            overlap_fn = create_loss_fn_pure(X_o, y_o, model, loss_fn, param_shapes)

            # Optimiser expects loss function handle if using line search
            if line_search_fn == "strong_wolfe":
                fn = create_loss_fn_pure(X_batch, y_batch, model, loss_fn, param_shapes)
            else:
                fn = None

            loss = optimizer.step(closure, overlap_fn, fn)
            epoch_loss += loss

        if scheduler is not None:
            scheduler.step()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)

        if math.isnan(epoch_loss):
            logger.info(f"\tLoss hit NaN at epoch {epoch+1}, stopping early")
            break

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn, device)
            test_losses.append(test_loss)
            for name, metric in metrics.items():
                metric_val = compute_metric(test_dataset, model, metric, device)
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
