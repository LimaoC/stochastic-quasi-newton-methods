from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from sqnm.optim.sqnhv import SQNHv

from .train_util import (
    compute_loss,
    create_closure,
    create_loss_fn_pure,
    log_training_info,
)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: SQNHv,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    generator: torch.Generator | None = None,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
    curvature_batch_size=600,
) -> dict[str, Any]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    skip = optimizer.param_groups[0]["skip"]
    line_search_fn = optimizer.param_groups[0]["line_search_fn"]

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(dataloader)

    curvature_sampler = BatchSampler(
        RandomSampler(train_dataset, replacement=True, generator=generator),
        curvature_batch_size,
        drop_last=False,
    )
    curvature_dataloader = DataLoader(train_dataset, batch_sampler=curvature_sampler)

    epoch_losses = []
    test_losses = []  # computed every log_frequency epochs
    k = 1
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)

            # Optimiser expects loss function handle if using line search
            if line_search_fn == "strong_wolfe":
                fn = create_loss_fn_pure(X_batch, y_batch, model, loss_fn, param_shapes)
            else:
                fn = None

            # Optimiser expects another function handle for computing curvature
            # estimates every skip epochs
            if k % skip == 0:
                X_c, y_c = next(iter(curvature_dataloader))
                X_c, y_c = X_c.to(device), y_c.to(device)
                curv_fn = create_loss_fn_pure(X_c, y_c, model, loss_fn, param_shapes)
            else:
                curv_fn = None

            epoch_loss += optimizer.step(closure, fn, curv_fn)
            k += 1

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


# def train_with_prob_ls(
#     train_dataset: TensorDataset,
#     test_dataset: TensorDataset,
#     optimizer: SQNHv,
#     model,
#     loss_fn,
#     ps_loss_fn,
#     device,
#     rng,
#     num_epochs=1000,
#     log_frequency=100,
#     batch_size=100,
#     curvature_batch_size=600,
# ) -> dict[str, Any]:
#     n = len(train_dataset)
#     params = {name: param.detach() for name, param in model.named_parameters()}
#     param_shapes = {name: param.shape for name, param in model.named_parameters()}
#
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     num_batches = len(train_dataloader)
#     # X_test, y_test = test_dataset[:]
#
#     skip = optimizer.param_groups[0]["skip"]
#
#     losses = []
#     grad_norms = []
#     k = 1
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         epoch_grad_norm = 0.0
#
#         for X_batch, y_batch in train_dataloader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#             closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
#             fn = create_loss_fn_with_vars_pure(
#                 X_batch, y_batch, model, loss_fn, ps_loss_fn, params, param_shapes
#             )
#
#             if k % skip == 0:
#                 idx = rng.choice(n, curvature_batch_size, replace=False)
#                 X_curv, y_curv = train_dataset[idx]
#                 X_curv, y_curv = X_curv.to(device), y_curv.to(device)
#
#                 curvature_fn = create_loss_fn_pure(
#                     X_curv, y_curv, model, loss_fn, param_shapes
#                 )
#                 loss = optimizer.step(closure, fn, curvature_fn)
#             else:
#                 loss = optimizer.step(closure, fn)
#
#             epoch_loss += loss
#             epoch_grad_norm += grad_vec(model.parameters()).norm()
#             k += 1
#
#         # Aggregate loss and grad norm across all batches in this epoch
#         epoch_loss /= num_batches
#         epoch_grad_norm /= num_batches
#         losses.append(epoch_loss)
#         grad_norms.append(epoch_grad_norm)
#
#         if epoch % log_frequency == log_frequency - 1:
#             with torch.no_grad():
#                 training_loss = compute_loss(train_dataset, model, loss_fn)
#             log_training_info(epoch + 1, training_loss, epoch_grad_norm)
#
#     return {"losses": losses, "grad_norms": grad_norms}
