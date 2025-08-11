from torch.utils.data import DataLoader, TensorDataset

from sqnm.optim.sqn_hv import SQNHv
from sqnm.utils.param import grad_vec

from .train_util import (
    create_closure,
    create_loss_fn_pure,
    create_loss_fn_with_vars_pure,
    log_training_info,
)


def train(
    X,
    y,
    optimizer: SQNHv,
    model,
    loss_fn,
    device,
    rng,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
    curvature_batch_size=600,
    skip=10,
) -> tuple[list[float], list[float]]:
    n, _ = X.shape
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dataloader)

    losses = []
    grad_norms = []
    k = 1
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)

            if k % skip == 0:
                idx = rng.choice(n, curvature_batch_size, replace=False)
                X_curv, y_curv = X[idx].to(device), y[idx].to(device)

                curvature_fn = create_loss_fn_pure(
                    X_curv, y_curv, model, loss_fn, param_shapes
                )
                loss = optimizer.step(closure, curvature_fn=curvature_fn)
            else:
                loss = optimizer.step(closure)

            epoch_loss += loss
            epoch_grad_norm += grad_vec(model.parameters()).norm()
            k += 1

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_grad_norm /= num_batches
        losses.append(epoch_loss)
        grad_norms.append(epoch_grad_norm)

        if epoch % log_frequency == log_frequency - 1:
            log_training_info(epoch + 1, epoch_loss, epoch_grad_norm)

    return losses, grad_norms


def train_with_prob_ls(
    X,
    y,
    optimizer: SQNHv,
    model,
    loss_fn,
    ps_loss_fn,
    device,
    rng,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
    curvature_batch_size=600,
    skip=10,
) -> tuple[list[float], list[float]]:
    n, _ = X.shape
    params = {name: param.detach() for name, param in model.named_parameters()}
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dataloader)

    losses = []
    grad_norms = []
    k = 1
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
            fn = create_loss_fn_with_vars_pure(
                X_batch, y_batch, model, loss_fn, ps_loss_fn, params, param_shapes
            )

            if k % skip == 0:
                idx = rng.choice(n, curvature_batch_size, replace=False)
                X_curv, y_curv = X[idx].to(device), y[idx].to(device)

                curvature_fn = create_loss_fn_pure(
                    X_curv, y_curv, model, loss_fn, param_shapes
                )
                loss = optimizer.step(closure, fn, curvature_fn)
            else:
                loss = optimizer.step(closure, fn)

            epoch_loss += loss
            epoch_grad_norm += grad_vec(model.parameters()).norm()
            k += 1

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_grad_norm /= num_batches
        losses.append(epoch_loss)
        grad_norms.append(epoch_grad_norm)

        if epoch % log_frequency == log_frequency - 1:
            log_training_info(epoch + 1, epoch_loss, epoch_grad_norm)

    return losses, grad_norms
