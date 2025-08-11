from sqnm.optim.lbfgs import LBFGS
from sqnm.utils.param import grad_vec

from .train_util import create_closure, create_loss_fn_pure, log_training_info


def train(
    X,
    y,
    optimizer: LBFGS,
    model,
    loss_fn,
    device,
    num_epochs=1000,
    log_frequency=100,
) -> tuple[list[float], list[float]]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

    X, y = X.to(device), y.to(device)
    closure = create_closure(X, y, optimizer, model, loss_fn)
    fn = create_loss_fn_pure(X, y, model, loss_fn, param_shapes)

    losses = []
    grad_norms = []
    for epoch in range(num_epochs):
        loss = optimizer.step(closure, fn)
        grad_norm = grad_vec(model.parameters()).norm()
        losses.append(loss)
        grad_norms.append(grad_norm)

        if epoch % log_frequency == log_frequency - 1:
            log_training_info(epoch + 1, loss, grad_norm)

    return losses, grad_norms
