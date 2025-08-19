from typing import Any

from torch.utils.data import Dataset

from sqnm.optim.lbfgs import LBFGS

from .train_util import (
    compute_loss,
    create_closure,
    create_loss_fn_pure,
    log_training_info,
)


def train(
    X,
    y,
    test_dataset: Dataset,
    optimizer: LBFGS,
    model,
    loss_fn,
    device,
    num_epochs=1000,
    log_frequency=100,
) -> dict[str, Any]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    X, y = X.to(device), y.to(device)
    closure = create_closure(X, y, optimizer, model, loss_fn)
    fn = create_loss_fn_pure(X, y, model, loss_fn, param_shapes)

    losses = []
    test_losses = []
    for epoch in range(num_epochs):
        loss = optimizer.step(closure, fn)
        losses.append(loss)

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn)
            test_losses.append(test_loss)
            log_training_info(epoch + 1, loss, test_loss)

    return {"epoch_losses": losses, "test_losses": test_losses}
