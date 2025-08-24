from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from sqnm.optim.scbfgs import SCBFGS

from .train_util import (
    compute_loss,
    create_closure,
    create_loss_fn_pure,
    log_training_info,
)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: SCBFGS,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    generator: torch.Generator | None = None,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
) -> dict[str, Any]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    line_search_fn = optimizer.param_groups[0]["line_search_fn"]

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    num_batches = len(dataloader)

    proxy_sampler = BatchSampler(
        RandomSampler(train_dataset, replacement=True, generator=generator),
        batch_size,
        drop_last=False,
    )
    proxy_dataloader = DataLoader(train_dataset, batch_sampler=proxy_sampler)

    epoch_losses = []
    test_losses = []  # computed every log_frequency epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)

            X_p, y_p = next(iter(proxy_dataloader))
            X_p, y_p = X_p.to(device), y_p.to(device)
            proxy_fn = create_loss_fn_pure(X_p, y_p, model, loss_fn, param_shapes)

            # Optimiser expects loss function handle if using line search
            if line_search_fn == "strong_wolfe":
                fn = create_loss_fn_pure(X_batch, y_batch, model, loss_fn, param_shapes)
            else:
                fn = None

            epoch_loss += optimizer.step(closure, proxy_fn, fn)

        if scheduler is not None:
            scheduler.step()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn)
            test_losses.append(test_loss)
            log_training_info(epoch + 1, epoch_loss, test_loss)

    return {"epoch_losses": epoch_losses, "test_losses": test_losses}
