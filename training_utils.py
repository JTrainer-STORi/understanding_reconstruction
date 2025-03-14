import jax
import jax.numpy as jnp
import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms
import neural_tangents as nt

import numpy as np
import flax
import flax.linen as nn
import optax as tx
import neural_tangents.stax as stax

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from typing import Any, Callable, Sequence, Tuple
from flax.training import train_state, checkpoints

import matplotlib.pyplot as plt
import functools
import operator
import fire

import data
from utils import _sub, multiply_by_scalar, get_dot_product, bind, _add, _zeros_like
import models

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'use_base_params', 'xent'))
def get_training_loss_l2(params, images, labels, net_train_state, l2 = 0., train = False, has_bn = False, batch_stats = None, use_base_params = False, label_scale = 1, init_params = None, xent = False):
    # outputs = net_forward_apply(params, images, train = False)['fc'][:, :10]


    variables = {'params': params}

    if init_params is not None:
        variables_init = {'params': init_params}

    if has_bn:
        variables['batch_stats'] = batch_stats

    # if net_train_state.base_params is not None:
    #     variables['base_params'] = net_train_state.base_params 

    if has_bn:
        outputs, new_batch_stats = net_train_state.apply_fn(variables, images, train = train, mutable=['batch_stats'])
        new_batch_stats = new_batch_stats['batch_stats']
    else:
        outputs, _ = net_train_state.apply_fn(variables, images, train = train, mutable=[])
        new_batch_stats = None

    # loss = jnp.sum(0.5 * (outputs - labels)**2)

    # loss = -jnp.sum(jax.nn.one_hot(labels) * jax.nn.log_softmax(outputs, 1))/len(labels)
    # loss = jnp.mean(bce(logits = outputs, labels = labels.astype(jnp.float32)))
    
    if not xent:
        loss = jnp.mean(0.5 * ((outputs - labels)) **2)
    else:
        if labels.shape[-1] == 1:
            loss = jnp.mean(tx.sigmoid_binary_cross_entropy(outputs, (labels / 2 ) + 1))
        else:
            loss = jnp.mean(tx.softmax_cross_entropy(outputs, labels + jnp.min(labels)))
    # loss = jnp.sum(0.5 * (outputs - labels)**2, axis = -1).mean()
    # loss = jnp.mean(0.5 * (outputs - labels)**2, axis = -1).sum()
    
    # if type(l2) is dict:
    #     loss += 0.5 * l2['body'] * get_dot_product(params, params)
    #     # head = params['tangent_params']['kernel']
    # else:
    #     if 'base_params' in params:
    #         loss += 0.5 * l2 * get_dot_product(params['tangent_params'], params['tangent_params'])
    #     else:
    #         loss += 0.5 * l2 * get_dot_product(params, params)


    # acc = jnp.mean(outputs.argmax(1) == labels.argmax(1))
    # acc = 0
    # acc = jnp.mean((outputs > 0).reshape(-1) == labels > 0.5)
    
    if labels.shape[-1] == 1:
        acc = jnp.mean((outputs > 0) == (labels > 0))
        n_correct = jnp.sum((outputs > 0) == (labels > 0))
    else:
        acc = jnp.mean(outputs.argmax(1) == labels.argmax(1))
        n_correct = jnp.sum(outputs.argmax(1) == labels.argmax(1))

    err = labels - outputs
    # n_correct = jnp.sum(outputs.argmax(1) == labels.argmax(1))
    
    # loss = loss/(labels.shape[0] * labels.shape[1])
    # loss = loss

    
    

    return loss, [new_batch_stats, acc, n_correct, err]

def custom_sigmoid_binary_cross_entropy(logits, labels):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p

def bce(logits, labels):
    
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    # print(labels.shape)
    # print(log_p.shape)
    labels = labels.reshape(-1, 1)
    return -labels * log_p - (1. - labels) * log_not_p

@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params', 'use_dp', 'xent'))
def do_training_step(train_state, training_batch, l2 = 0., has_bn = False, train = True, update_ema = False, ema_decay = 0.995, use_base_params = False, use_dp = False, label_scale = None, init_params = None, xent = False):
    images = training_batch['images']
    labels = training_batch['labels']
    
    if has_bn:
        batch_stats = train_state.batch_stats
    else:
        batch_stats = None

    # get_training_loss_l2(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params)
    
    if use_dp:
        # loss, (new_batch_stats, acc, _) = get_training_loss_l2(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params)
        (loss, (_, acc, _)), grad = jax.vmap(jax.value_and_grad(bind(get_training_loss_l2, ..., ..., ..., train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params), has_aux = True), in_axes = [None, 0, 0])(train_state.params, images[:, None], labels[:, None])
        loss = jnp.mean(loss)
        acc = jnp.mean(acc)
    else:
        (loss, (new_batch_stats, acc, _, err)), grad = jax.value_and_grad(get_training_loss_l2, argnums = 0, has_aux = True)(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params, label_scale = label_scale)    
        
        
    if has_bn:
        new_state = train_state.apply_gradients(grads = grad, batch_stats = new_batch_stats, train_it = train_state.train_it + 1)
    else:
        new_state = train_state.apply_gradients(grads = grad, train_it = train_state.train_it + 1)
    
    if update_ema:
        new_ema_hidden, new_ema_average = get_updated_ema(new_state.params, new_state.ema_hidden, ema_decay, new_state.train_it, order = 1)
        new_state = new_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)
    
    return new_state, (loss, grad, err)


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    train_it: int
    ema_hidden: Any = None
    ema_average: Any = None
    base_params: Any = None

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'use_base_params', 'use_softplus', 'mean'))
def get_g_value(image_train_state, net_params, image_params, dual_params, has_bn = False, use_base_params = False, use_softplus = True, train = False, batch_stats = None, beta = 1., mean = True):
    def fwd(net_params, dual_params):
        if has_bn:
            variables = {'params': net_params, 'batch_stats': batch_stats}
        else:
            variables = {'params': net_params}
            
        # if net_train_state.base_params is not None:
        #     variables['base_params'] = net_train_state.base_params 

        # print(batch_stats)
        # print(has_bn)
        # print(train)

        if use_base_params:
            outputs, new_batch_stats = image_train_state.apply_fn(variables, image_params, train = train, mutable=['batch_stats'], use_base_params = use_base_params, use_softplus = use_softplus, beta = beta)
        else:
            outputs, new_batch_stats = image_train_state.apply_fn(variables, image_params, train = train, mutable=['batch_stats'], use_softplus = use_softplus, beta = beta)
        
        
        
        # print(labels_fixed.shape)
        # print(outputs.shape)
        if mean:
            return jnp.mean(dual_params * outputs)
        else:
            return jnp.mean(dual_params * outputs) * dual_params.shape[0]
    
    return jax.grad(fwd)(net_params, dual_params)

@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'use_base_params', 'use_softplus', 'both_kernels'))
def get_recon_loss(all_params, init_params, net_params, labels, image_train_state, l2 = 0., train = False, has_bn = False, batch_stats = None, use_base_params = False, beta = 1., use_softplus = True, img_min = 0, img_max = 1, both_kernels = False, high_freq_coef = 1, sparsity = 0.0):
    # outputs = net_forward_apply(params, images, train = False)['fc'][:, :10]6
    
    # net_params = multiply_by_scalar(net_params, 1/jnp.sqrt(get_dot_product(net_params, net_params)))
    
    image_params = all_params['images']
    # image_params = jnp.concatenate([train_images for i in range(amp_factor)], 0)
    dual_params = all_params['duals']
    dual_params2 = all_params['duals2']
    
    stupid_init_params = init_params
    
    labels_fixed = labels
    labels_fixed = labels_fixed - image_train_state.apply_fn({'params': stupid_init_params}, image_params, train = train, mutable=['batch_stats'], use_softplus = use_softplus, beta = beta)[0]
    

    grad = get_g_value(image_train_state, net_params, image_params, dual_params, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, )#jax.grad(fwd, has_aux = True)(net_params, dual_params)

    if both_kernels:
        grad2 = get_g_value(image_train_state, stupid_init_params, image_params, dual_params2, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, )#jax.grad(fwd, has_aux = True)(net_params, dual_params)
    # grad2, _ = jax.grad(fwd, has_aux = True)(stupid_init_params, dual_params2)

        grad = _add(grad, grad2)
    
    
    delta_params = _sub(net_params, stupid_init_params)
    
    
    diff = _sub(multiply_by_scalar(delta_params, 1), grad)

    stationary_loss = get_dot_product(diff, diff)
    
    
    img_loss = (jnp.mean(jax.nn.relu(image_params - img_max[None, None, None])**2 + jax.nn.relu(- image_params + img_min[None, None, None])**2)) * image_params.shape[0]

    blurred = blur_batch(image_params)
    # print(blurred)
    high_freq = image_params - blurred
    
    high_freq_loss = 0.0015 * jnp.mean(high_freq ** 2) * image_params.shape[0]
    high_freq_loss = high_freq_coef * high_freq_loss
    sparse_loss = sparsity * (jnp.mean(jnp.abs(dual_params)) + jnp.mean(jnp.abs(dual_params2)))
    
    loss = stationary_loss + 0.1 * img_loss + high_freq_loss + sparse_loss#+ eig_loss 

    
    return loss, (None, None ,0)


@functools.partial(jax.jit, static_argnames=('train', 'has_bn', 'use_base_params', 'use_softplus', 'both_kernels'))
def get_recon_loss_batched(image_params, dual_params, dual_params2, init_params, net_params, image_train_state, l2 = 0., train = False, has_bn = False, batch_stats = None, use_base_params = False, beta = 1., use_softplus = True, img_min = 0, img_max = 1, g_state = None, n_images = 1, both_kernels = False, high_freq_coef = 1):
    stupid_init_params = init_params    
    


    grad_batch = get_g_value(image_train_state, net_params, image_params, dual_params, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
    if both_kernels:
        grad_batch2 = get_g_value(image_train_state, init_params, image_params, dual_params2, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
        grad_batch = _add(grad_batch, grad_batch2)
    # grad2, _ = jax.grad(fwd, has_aux = True)(stupid_init_params, dual_params2)

    # grad = _add(grad, grad2)

    grad = _add(_sub(g_state, jax.lax.stop_gradient(grad_batch)), grad_batch)
    # print(_sub(g_state, jax.lax.stop_gradient(grad_batch)))
    
    grad = multiply_by_scalar(grad, 1/n_images)
    
    delta_params = _sub(net_params, stupid_init_params)
    
    
    diff = _sub(multiply_by_scalar(delta_params, 1), grad)

    stationary_loss = get_dot_product(diff, diff)
    
    
    
    img_loss = (jnp.mean(jax.nn.relu(image_params - img_max[None, None, None])**2 + jax.nn.relu(- image_params + img_min[None, None, None])**2)) * image_params.shape[0]

    blurred = blur_batch(image_params)
    # print(blurred)
    high_freq = image_params - blurred
    

    high_freq_loss = 0.0015 * jnp.mean(high_freq ** 2) * image_params.shape[0]
    high_freq_loss = high_freq_coef * high_freq_loss
    
    loss = stationary_loss + 0.1 * img_loss + high_freq_loss

    
    return loss, grad_batch

def blur_batch(images):
    filt = jnp.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

    def blur_single(img):
        return jax.scipy.signal.convolve2d(img, filt, mode = 'same', boundary = 'fill')
    
    return jax.vmap(jax.vmap(blur_single, in_axes = (2, ), out_axes=2), in_axes = (0, ))(images)

@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params',  'use_softplus', 'both_kernels'))
def do_training_step_recon(train_state, labels, init_params, final_params, l2 = 0., has_bn = False, train = False, update_ema = False, ema_decay = 0.995, use_base_params = False, beta = 1., use_softplus = True, img_min = 0, img_max = 1, batch_stats = None, both_kernels = False, high_freq_coef = 1, sparsity = 0.0):

    # get_training_loss_l2(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params)
        
    (loss, (_, acc, _)), grad = jax.value_and_grad(get_recon_loss, argnums = 0, has_aux = True)(train_state.params, init_params, final_params, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params, beta = beta, use_softplus = use_softplus, 
    img_min = img_min, img_max = img_max, both_kernels = both_kernels, high_freq_coef = high_freq_coef, sparsity = sparsity)
    
#     iters_per = 8000
#     index_to_update = (train_state.train_it % (iters_per * train_state.params['images'].shape[0]))//iters_per
    
#     grad_mask = jnp.zeros(shape = [train_state.params['images'].shape[0]])
#     grad_mask = grad_mask.at[index_to_update].set(1.)
    
#     grad['duals'] = grad['duals'] * grad_mask
#     grad['images'] = grad['images'] * grad_mask[:, None, None, None]

    new_state = train_state.apply_gradients(grads = grad, train_it = train_state.train_it + 1)
    
    if update_ema:
        new_ema_hidden, new_ema_average = get_updated_ema(new_state.params, new_state.ema_hidden, ema_decay, new_state.train_it, order = 1)
        new_state = new_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)
    
    return new_state, (loss, acc)


@functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params',  'use_softplus', 'both_kernels'))
def do_training_step_recon_batched(train_state, labels, init_params, final_params, l2 = 0., has_bn = False, train = False, update_ema = False, ema_decay = 0.995, use_base_params = False, beta = 1., use_softplus = True, img_min = 0, img_max = 1, batch_size = -1, g_state = None, key = jax.random.PRNGKey(0), both_kernels = False, batch_stats = None, high_freq_coef = 1):

    # get_training_loss_l2(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params)

    n_total_recon = train_state.params['images'].shape[0]
    batch_indices = jax.random.choice(key, n_total_recon, shape = (batch_size,), replace = False)

    image_batch = train_state.params['images'][batch_indices]
    dual_batch = train_state.params['duals'][batch_indices]
    dual_batch2 = train_state.params['duals2'][batch_indices]
        
    (loss, grad_batch), (image_grads, dual_grads, dual_grads2) = jax.value_and_grad(get_recon_loss_batched, argnums = [0, 1, 2], has_aux = True)(image_batch, dual_batch, dual_batch2, init_params, final_params, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params, beta = beta, use_softplus = use_softplus, 
    img_min = img_min, img_max = img_max, g_state = g_state, n_images = n_total_recon, both_kernels = both_kernels, high_freq_coef = high_freq_coef)
    
    full_image_grads = jnp.zeros_like(train_state.params['images'])
    full_dual_grads = jnp.zeros_like(train_state.params['duals'])
    full_dual_grads2 = jnp.zeros_like(train_state.params['duals2'])

    full_image_grads = full_image_grads.at[batch_indices].set(image_grads)
    full_dual_grads = full_dual_grads.at[batch_indices].set(dual_grads)
    full_dual_grads2 = full_dual_grads2.at[batch_indices].set(dual_grads2)

    full_grads = {
        'images': full_image_grads,
        'duals': full_dual_grads,
        'duals2': full_dual_grads2
    }
        
    # if has_bn:
    #     new_state = train_state.apply_gradients(grads = full_grads, batch_stats = new_batch_stats['batch_stats'], train_it = train_state.train_it + 1)
    # else:
    new_state = train_state.apply_gradients(grads = full_grads, train_it = train_state.train_it + 1)

    # new_state = train_state
    
    if update_ema:
        new_ema_hidden, new_ema_average = get_updated_ema(new_state.params, new_state.ema_hidden, ema_decay, new_state.train_it, order = 1)
        new_state = new_state.replace(ema_average = new_ema_average, ema_hidden = new_ema_hidden)

    updated_image_batch = new_state.params['images'][batch_indices]
    updated_dual_batch = new_state.params['duals'][batch_indices]
    updated_dual_batch2 = new_state.params['duals2'][batch_indices]

    new_grad_batch = get_g_value(train_state, final_params, updated_image_batch, updated_dual_batch, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
    if both_kernels:
        new_grad_batch2 = get_g_value(train_state, init_params, updated_image_batch, updated_dual_batch2, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
        new_grad_batch = _add(new_grad_batch, new_grad_batch2)
    
    # print(_sub(g_state, grad_batch))
    g_state = _add(_sub(g_state, grad_batch), new_grad_batch)
    # print("BONGER")
    # print(_sub(g_state, grad_batch))
    
    return new_state, (loss, g_state, jax.random.split(key)[0])
    # return new_state, (loss, g_state, key)



# @functools.partial(jax.jit, static_argnames=('has_bn', 'train', 'update_ema', 'use_base_params',  'use_softplus'))
# def do_training_step_recon(train_state, training_batch, direction, net_params, l2 = 0., has_bn = False, train = True, update_ema = False, ema_decay = 0.995, use_base_params = False, use_softplus = True, n_steps = 10000):
    
#     def body_fn(i, val):
#         train

def init_g_state(train_state, init_params, final_params, has_bn = False, train = False, use_base_params = False, beta = 1., use_softplus = True, img_min = 0, img_max = 1, batch_size = -1, both_kernels = False, batch_stats = None):
    g_state = _zeros_like(init_params)

    n_images = train_state.params['images'].shape[0]

    for b in range(int(np.ceil(n_images/batch_size))):
        image_batch = train_state.params['images'][b * batch_size: (b+1) * batch_size]
        dual_batch = train_state.params['duals'][b * batch_size: (b+1) * batch_size]
        dual_batch2 = train_state.params['duals2'][b * batch_size: (b+1) * batch_size]

        # print(image_batch.shape)

        g_batch = get_g_value(train_state, final_params, image_batch, dual_batch, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
        if both_kernels:
            g_batch2 = get_g_value(train_state, init_params, image_batch, dual_batch2, has_bn = has_bn, use_base_params = use_base_params, use_softplus = use_softplus, train = train, batch_stats = batch_stats, beta = beta, mean = False)
            g_batch = _add(g_batch, g_batch2)


        g_state = _add(g_state, g_batch)

    return g_state
