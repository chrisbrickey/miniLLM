from collections.abc import Callable

import flax.nnx as nnx
import jax.numpy as jnp

from src.loss import cross_entropy_loss


def make_train_step() -> Callable[
    [nnx.Module, nnx.ModelAndOptimizer, nnx.MultiMetric, tuple[jnp.ndarray, jnp.ndarray]],
    None,
]:
    """JIT-compiled training step factory that returns a compiled function,
    specifically an @nnx.jit-compiled train_step function.

    Calling the returned function will execute a single gradient update
    for the batch passed in as a parameter to the factory.

    NB: Returning a factory keeps the @nnx.jit decoration explicit and makes it easy
    to swap in a non-jit version for debugging or a pmap version for multi-device.
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.ModelAndOptimizer,
        metrics: nnx.MultiMetric,
        batch: tuple[jnp.ndarray, jnp.ndarray],
    ) -> None:
        grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
        (loss, _logits), grads = grad_fn(model, batch)
        metrics.update(loss=loss)
        optimizer.update(grads)

    return train_step
