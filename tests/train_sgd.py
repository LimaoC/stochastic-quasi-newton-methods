from typing import Any

import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from .train_util import compute_loss, create_closure, log_training_info


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: SGD,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
) -> dict[str, Any]:
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)

    epoch_losses = []
    test_losses = []  # computed every log_frequency epochs
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

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn)
            test_losses.append(test_loss)
            log_training_info(epoch + 1, epoch_loss, test_loss)

    return {"epoch_losses": epoch_losses, "test_losses": test_losses}
