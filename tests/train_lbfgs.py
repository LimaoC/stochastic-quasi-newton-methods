from sqnm.utils.param import grad_vec

from .train_util import (
    compute_acc,
    create_closure,
    create_loss_fn_pure,
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
) -> tuple[list[float], list[float]]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}

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
            acc = compute_acc(X, y, model)
            log_training_info(epoch + 1, loss, acc, grad_norm)

    return losses, grad_norms
