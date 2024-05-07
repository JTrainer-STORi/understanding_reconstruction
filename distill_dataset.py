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
from utils import *
import models
import training_utils
import pickle


def get_noise_mask(key, shape, p):
    return (jax.random.uniform(key, shape = shape) > p).astype(jnp.float32)

def mask_noise(images, noise, noise_mask):
    return images * noise_mask + noise * (1. - noise_mask)

@functools.partial(jax.jit, static_argnames=('kernel_fn', 'use_mse', 'learn_labels'))
def get_distillation_loss(distill_train_state_params, train_images, train_labels, kernel_fn, noise_mask = None, init_noise = None, use_mse = True, learn_labels = False, minimum_dist = -1):
    distill_images = distill_train_state_params['images']
    log_temp = distill_train_state_params['log_temp']
    distilled_labels = distill_train_state_params['labels']
    if not learn_labels:
        distilled_labels = jax.lax.stop_gradient(distilled_labels)
    
    log_temp = jnp.minimum(log_temp, 8)
    
    if noise_mask is not None:
        distill_images = mask_noise(distill_images, init_noise, noise_mask)
    
    K_ss = kernel_fn(distill_images, distill_images, 'ntk')
    K_ts = kernel_fn(train_images, distill_images, 'ntk')

    
    K_ss = K_ss + 1e-5 * jnp.trace(K_ss) * jnp.eye(K_ss.shape[0])/K_ss.shape[0]
    
    preds = K_ts @ jnp.linalg.solve(K_ss, distilled_labels)
    # print((train_labels - preds).shape)
    
    
    if train_labels.shape[-1] == 1:
        if not use_mse:
            loss = jnp.mean(jax.nn.log_sigmoid(-1 * train_labels * preds * jnp.exp(log_temp))) #+ 0.01 * jnp.exp(log_temp)**2
        else:
            loss = 0.5 * jnp.mean((train_labels - preds) ** 2)


    train_flattened = train_images.reshape(train_images.shape[0], -1)
    distilled_images_clipped = jnp.clip(distill_images, jnp.min(train_images, axis = [0, 1, 2], keepdims = True), jnp.max(train_images, axis = [0, 1, 2], keepdims = True))
    distilled_flattened = distilled_images_clipped.reshape(distilled_images_clipped.shape[0], -1)
    

    distances = jnp.sum((distilled_flattened[:, None] - train_flattened[None])**2, -1)
    
    min_dists = jnp.min(distances, 1)

    dist_loss = (minimum_dist - jax.numpy.minimum(min_dists, minimum_dist))**2

    loss += 50 * jnp.mean(dist_loss)
    
    return loss, 0


@functools.partial(jax.jit, static_argnames=('kernel_fn', 'use_mse', 'learn_labels'))
def get_distillation_loss_rkip(distill_train_state_params, train_images, train_labels, kernel_fn, noise_mask = None, init_noise = None, use_mse = False, learn_labels = False, minimum_dist = -1):
    distill_images = distill_train_state_params['images']
    log_temp = distill_train_state_params['log_temp']
    distilled_labels = distill_train_state_params['labels']

    if not learn_labels:
        distilled_labels = jax.lax.stop_gradient(distilled_labels)
    
    log_temp = jnp.minimum(log_temp, 8)
    
    if noise_mask is not None:
        distill_images = mask_noise(distill_images, init_noise, noise_mask)
    
    K_ss = kernel_fn(distill_images, distill_images, 'ntk')
    K_ts = kernel_fn(train_images, distill_images, 'ntk')
    K_tt = kernel_fn(train_images, train_images, 'ntk')

    
    K_ss_reg = K_ss + 1e-5 * jnp.trace(K_ss) * jnp.eye(K_ss.shape[0])/K_ss.shape[0]
    K_tt_reg = K_tt + 1e-5 * jnp.trace(K_tt) * jnp.eye(K_tt.shape[0])/K_tt.shape[0]
    
    alpha_s = jnp.linalg.solve(K_ss_reg, distilled_labels)

    alpha_t = jnp.linalg.solve(K_tt_reg, train_labels)

    t1 = alpha_s.T @ K_ss @ alpha_s
    t2 = -2 * alpha_t.T @ K_ts @ alpha_s
    t3 = alpha_t.T @ K_tt @ alpha_t

    # print((train_labels - preds).shape)
    

    loss = 0.5 * jnp.sum(t1 + t2 + t3)

    train_flattened = train_images.reshape(train_images.shape[0], -1)
    distilled_images_clipped = jnp.clip(distill_images, jnp.min(train_images, axis = [0, 1, 2], keepdims = True), jnp.max(train_images, axis = [0, 1, 2], keepdims = True))
    distilled_flattened = distilled_images_clipped.reshape(distilled_images_clipped.shape[0], -1)

    distances = jnp.sum((distilled_flattened[:, None] - train_flattened[None])**2, -1)
    
    min_dists = jnp.min(distances, 1)

    dist_loss = (minimum_dist - jax.numpy.minimum(min_dists, minimum_dist))**2

    loss += 50 * jnp.mean(dist_loss)
    
    
    return loss, min_dists

@functools.partial(jax.jit, static_argnames=('kernel_fn', 'use_mse', 'rkip', 'learn_labels'))
def do_training_step_distillation(train_state, train_images, train_labels, kernel_fn, noise_mask = None, init_noise = None, use_mse = False, rkip = False, learn_labels = False, minimum_dist = -1):

    # get_training_loss_l2(train_state.params, images, labels, train_state, l2 = l2, train = train, has_bn = has_bn, batch_stats = batch_stats, use_base_params = use_base_params)
        
    if not rkip:
        (loss, acc), grad = jax.value_and_grad(get_distillation_loss, argnums = 0, has_aux = True)(train_state.params, train_images, train_labels, kernel_fn, noise_mask = noise_mask, init_noise = init_noise, use_mse = use_mse, learn_labels = learn_labels, minimum_dist = minimum_dist)
    else:
        (loss, acc), grad = jax.value_and_grad(get_distillation_loss_rkip, argnums = 0, has_aux = True)(train_state.params, train_images, train_labels, kernel_fn, noise_mask = noise_mask, init_noise = init_noise, use_mse = use_mse, learn_labels = learn_labels, minimum_dist = minimum_dist)
    
    new_state = train_state.apply_gradients(grads = grad, train_it = train_state.train_it + 1)
    
    return new_state, (loss, acc)


def main(seed = 0, dataset_name = 'mnist_odd_even', output_dir = None, train_set_size = 100, n_per_class_distilled = 1, noise_ratio = None, use_mse = False, rkip = False, learn_labels = False, minimum_dist = -1):
    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(locals()))

    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)

    key = jax.random.PRNGKey(seed)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Flatten(),
        stax.Dense(2048, W_std = 1, parameterization = 'ntk'),
        stax.Relu(),
        stax.Dense(2048, W_std = 1, parameterization = 'ntk'),
        stax.Relu(),
        stax.Dense(1, W_std = 1, parameterization = 'ntk')
    )

    distilled_labels_init = (jnp.array([-1 for i in range(n_per_class_distilled)] + [1 for i in range(n_per_class_distilled)]).astype(jnp.float32)).reshape(-1, 1)

    init_images = {
        'images': 0.2 * jnp.array(jax.random.normal(key, shape = [n_per_class_distilled * 2, train_images.shape[1], train_images.shape[2], train_images.shape[3]])),
        'log_temp': 2.3 * jnp.ones(()),
        'labels': distilled_labels_init
    }

    key = jax.random.split(key)[0]

    

    if noise_ratio is not None:
        init_noise = jax.random.normal(key, shape = init_images['images'].shape)

        key = jax.random.split(key)[0]

        noise_mask = get_noise_mask(key, shape = init_images['images'].shape, p = noise_ratio)
    else:
        init_noise = None,
        noise_mask = None

    
    opt = tx.chain(tx.adam(learning_rate=0.001))

    distill_train_state = training_utils.TrainStateWithBatchStats.create(apply_fn = None, params = init_images, tx = opt, batch_stats = None, train_it = 0, base_params = None)
    
    max_iters = 50000
    for i in range(max_iters + 1):
        distill_train_state, (loss, acc) = do_training_step_distillation(distill_train_state, train_images, train_labels, kernel_fn, noise_mask = noise_mask, init_noise = init_noise, use_mse = use_mse, rkip = rkip, learn_labels = learn_labels, minimum_dist = minimum_dist)

        if i % 10000 == 0:
            print(f'iter: {i}, distill loss: {loss}')


    distilled_images = distill_train_state.params['images']
    distilled_labels = distill_train_state.params['labels']

    if noise_ratio is not None:
        distilled_images = mask_noise(distill_train_state.params['images'], init_noise, noise_mask)

    

    output_dict = {
        'train_images': train_images,
        'train_labels': train_labels,
        'train_mean': train_mean,
        'distilled_images': distilled_images,
        'log_temp': distill_train_state.params['log_temp'],
        'distilled_labels': distilled_labels,
        'distillation_loss': loss
    }

    

    if output_dir is not None:
        pickle.dump(output_dict, open('./{}/distillation_result.pkl'.format(output_dir), 'wb'))

    print('done')

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)