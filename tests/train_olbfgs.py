from torch.utils.data import DataLoader, TensorDataset

from sqnm.utils.param import grad_vec

from .train_util import (
    compute_acc,
    create_closure,
    create_loss_fn_with_vars_pure,
    log_training_info,
)


def train(
    X,
    y,
    optimizer,
    model,
    loss_fn,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
) -> tuple[list[float], list[float]]:
    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dataloader)

    losses = []
    grad_norms = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for X_batch, y_batch in train_dataloader:
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
            epoch_loss += optimizer.step(closure)
            epoch_grad_norm += grad_vec(model.parameters()).norm()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_grad_norm /= num_batches
        losses.append(epoch_loss)
        grad_norms.append(epoch_grad_norm)

        if epoch % log_frequency == log_frequency - 1:
            acc = compute_acc(X, y, model)
            log_training_info(epoch + 1, epoch_loss, acc, epoch_grad_norm)

    return losses, grad_norms


def train_with_prob_ls(
    X,
    y,
    optimizer,
    model,
    loss_fn,
    ps_loss_fn,
    num_epochs=1000,
    batch_size=100,
    log_frequency=100,
) -> tuple[list[float], list[float]]:
    params = {name: param.detach() for name, param in model.named_parameters()}
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    train_dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dataloader)

    losses = []
    grad_norms = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for X_batch, y_batch in train_dataloader:
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
            fn = create_loss_fn_with_vars_pure(
                X_batch, y_batch, model, loss_fn, ps_loss_fn, params, param_shapes
            )
            epoch_loss += optimizer.step(closure, fn)
            epoch_grad_norm += grad_vec(model.parameters()).norm()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_grad_norm /= num_batches
        losses.append(epoch_loss)
        grad_norms.append(epoch_grad_norm)

        if epoch % log_frequency == log_frequency - 1:
            acc = compute_acc(X, y, model)
            log_training_info(epoch + 1, epoch_loss, acc, epoch_grad_norm)

    return losses, grad_norms
