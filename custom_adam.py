import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import clipping
from optax._src import numerics
from optax._src import utils
from optax._src import wrappers

from utils import _add, _sub, _multiply, _ones_like, _pow

# import utils

ScalarOrSchedule = Union[float, jax.Array, base.Schedule]
from optax._src import combine
from optax._src import transform


_abs_sq = numerics.abs_sq

class ScaleByAdamStateSparse(NamedTuple):
  """State for the Adam algorithm."""
  count: base.Params
  mu: base.Updates
  nu: base.Updates

def update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)

def update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return _abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)

@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction = _sub(_ones_like(count), _pow(decay, count))

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
      lambda t, bc: t / bc.astype(t.dtype), moment, bias_correction)


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)

def scale_by_adam_sparse(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.
  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamStateSparse(count=jax.tree_util.tree_map(jnp.zeros_like, params), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    update_mask = jax.tree_util.tree_map(lambda x: (x != 0.).astype(x.dtype), updates)

    inv_mask = _sub(_ones_like(update_mask), update_mask)

    mu = _add(_multiply(inv_mask, state.mu), _multiply(update_mask, update_moment(updates, state.mu, b1, 1)))
    nu = _add(_multiply(inv_mask, state.nu), _multiply(update_mask, update_moment_per_elem_norm(updates, state.nu, b2, 2)))


    count_inc = _add(state.count, update_mask)

    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    

    updates =  _multiply(update_mask, updates)

    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamStateSparse(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def sparse_adam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  
  return combine.chain(
      scale_by_adam_sparse(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      _scale_by_learning_rate(learning_rate),
  )