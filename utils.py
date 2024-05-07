import jax
import jax.numpy as jnp
import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms
import neural_tangents as nt

import numpy as np

import functools
import operator

def _sub(x, y):
    return jax.tree_util.tree_map(operator.sub, x, y)
    
def _add(x, y):
    return jax.tree_util.tree_map(operator.add, x, y)

def _multiply(x, y):
    return jax.tree_util.tree_map(operator.mul, x, y)

def _pow(base, exp):
    return jax.tree_util.tree_map(lambda x: jnp.power(base, x), exp)

def get_dot_product(a, b):
    return jnp.sum(sum_tree(_multiply(a, b)))

def sum_reduce(a, b):
    return jnp.sum(a) + jnp.sum(b)

def sum_tree(x):
    return jax.tree_util.tree_reduce(sum_reduce , x)

def multiply_by_scalar(x, s):
    return jax.tree_util.tree_map(lambda x: s * x, x)

def get_cos(a, b):
    return get_dot_product(a,b)/jnp.sqrt(get_dot_product(a,a) * get_dot_product(b,b))

def make_noise_like(key, x):
    return jax.random.normal(key, x.shape)

def make_noise_like_tree(x, key):
    return jax.tree_util.tree_map(functools.partial(make_noise_like, key), x)

def copy_tree(x):
    return jax.tree_util.tree_map(lambda x: jnp.copy(x), x)

def _ones_like(x):
    return jax.tree_util.tree_map(lambda x: jnp.ones_like(x), x)

def _zeros_like(x):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), x)

def to_dtype(x, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), x)


class bind(functools.partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
    
def get_linear_forward(net_apply, base_params, batch_stats, has_bn = False):
    base_params = copy_tree(base_params)
    if not has_bn:
        def inner_fn(inner_params, images, **kwargs):
            return net_apply({'params': inner_params}, images, **kwargs)
        
        def linear_forward(variables_dict, images, **kwargs):
            primals, duals, aux = jax.jvp(bind(inner_fn, ..., images, **kwargs), (base_params,), (_sub(variables_dict['params'], base_params),), has_aux = True)
            
            return _add(primals, duals), aux
    else:
        def inner_fn(inner_params, images, **kwargs):
            return net_apply({'params': inner_params, 'batch_stats': batch_stats}, images, **kwargs)
        
        def linear_forward(variables_dict, images, **kwargs):
            primals, duals, aux = jax.jvp(bind(inner_fn, ..., images, **kwargs), (base_params,), (_sub(variables_dict['params'], base_params),), has_aux = True)
            
            return _add(primals, duals), aux
    
    return linear_forward


