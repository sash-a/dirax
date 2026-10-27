import jax
import optax


def mean_by_axis(axis_name: str) -> optax.GradientTransformation:
    """Optax transform that scales the grads by the size of the named axis.

    This is useful for data parallel training in jax, where jax will
    automatically psum the gradients, we then need to scale the gradients by
    the number of devices which is what this does.

    This is expected to be called inside a `shard_map`ed function where `check_vma`
    is true as this is where jax automatically psum's. If `check_vma` is false, then
    you would rather `pmean` here as jax will not automatically psum.

    Usage:
        optax.chain(scale_by_axis("device"), optax.adam(1e-4))

    Args:
        axis_name: the name of the device axis that params are sharded over.

    returns: an optax gradient transform that pmeans grads over a given axis.
    """

    def all_reduce(updates: optax.Updates, params: optax.Params | None) -> optax.Updates:
        return jax.tree.map(lambda g: g / jax.lax.axis_size(axis_name), updates)

    return optax.stateless(all_reduce)
