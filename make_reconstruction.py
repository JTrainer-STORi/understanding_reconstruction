import jax
import jax.numpy as jnp
import torch


# from jax.config import config
# config.update("jax_enable_x64", True)

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
import utils
from custom_adam import sparse_adam

def main(seed = 0, dataset_name = 'mnist_odd_even', model_width = 1000, train_set_size = 100, output_dir = None, linearize = False, checkpoint_name = 'final_checkpoint', max_iters = 80000, amp_factor = 2, ntk_param = False, no_bias = False, init_checkpoint_name = None, save_name = 'reconstruction', noise = 0.0, batch_size = -1, scale_iters = False, pretrained = False, both_kernels = False, sparsity = 0):
    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

    key = jax.random.PRNGKey(seed)
    key, image_key, dual_key = jax.random.split(key, 3)

    _, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)


    checkpoint_dict = pickle.load(open('./{}/{}.pkl'.format(output_dir, checkpoint_name), 'rb'))
    # real_images = pickle.load(open('./{}/{}.pkl'.format(output_dir, checkpoint_name), 'rb'))

    init_params = checkpoint_dict['init_params']
    final_params = checkpoint_dict['final_params']

    batch_stats = checkpoint_dict.get('batch_stats', None)

    if noise > 0.0:
        final_params = utils._add(checkpoint_dict['final_params'], multiply_by_scalar(make_noise_like_tree(checkpoint_dict['final_params'], jax.random.PRNGKey(seed)), noise))

    if init_checkpoint_name is not None:
        init_checkpoint_dict = pickle.load(open('./{}/{}.pkl'.format(output_dir, init_checkpoint_name), 'rb'))
        init_params = init_checkpoint_dict['final_params']

    # print(train_labels.shape)
    # print(init_params)
    # print(batch_stats)
    if pretrained:
        model = models.ResNet18(output_dim = train_labels.shape[-1])
        has_bn = True
    else:
        model = models.MLP(width = [model_width, model_width], ntk_param = ntk_param, no_bias = no_bias, output_dim = train_labels.shape[-1])
        has_bn = False
        
    net_init, net_apply_base = model.init, model.apply


    if linearize:
        net_apply = get_linear_forward(net_apply_base, init_params, batch_stats, has_bn = has_bn)
    else:
        net_apply = net_apply_base

    reconstruction_size = int(amp_factor * train_set_size)

    if 'mnist' in dataset_name:
        init_images = {
            'images': jnp.array(0.2 * jax.random.normal(image_key, shape = [reconstruction_size, 28, 28, 1])),
            'duals': jnp.array(jax.random.uniform(dual_key, shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
            'duals2': jnp.array(jax.random.uniform(jax.random.split(dual_key)[0], shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
        }
    elif 'cifar10' in dataset_name:
        train_images = pickle.load(open('./{}/{}.pkl'.format(output_dir, 'training_set'), 'rb'))['train_images']
        init_images = {
            'images': jnp.array(0.2 * jax.random.normal(image_key, shape = [reconstruction_size, 32, 32, 3])),
            # 'images': jnp.array(np.concatenate([train_images for i in range(amp_factor)], 0)),
            'duals': jnp.array(jax.random.uniform(dual_key, shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
            'duals2': jnp.array(jax.random.uniform(jax.random.split(dual_key)[0], shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
        }
    elif 'cub' in dataset_name:
        resolution = int(dataset_name[-2:])
        train_images = pickle.load(open('./{}/{}.pkl'.format(output_dir, 'training_set'), 'rb'))['train_images']
        init_images = {
            'images': jnp.array(0.2 * jax.random.normal(image_key, shape = [reconstruction_size, resolution, resolution, 3])),
            # 'images': jnp.array(np.concatenate([train_images for i in range(amp_factor)], 0)),
            'duals': jnp.array(jax.random.uniform(dual_key, shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
            'duals2': jnp.array(jax.random.uniform(jax.random.split(dual_key)[0], shape = [reconstruction_size, train_labels.shape[-1]])) - 0.5,
        }


    train_labels = (jnp.array([-1 for i in range(int(reconstruction_size//2))] + [1 for i in range(int(reconstruction_size//2))]).astype(jnp.float32)).reshape(-1, 1)

    # opt = tx.chain(tx.adam(learning_rate=0.02))
    
    opt = tx.chain(sparse_adam(learning_rate=0.02))
    # opt = tx.chain(sparse_adam(learning_rate=0.005))
    # opt = tx.masked(sparse_adam(learning_rate=0.005), {'images': False, 'duals': True, 'duals2': True})
    # opt = tx.chain(sparse_adam(learning_rate=0.001))

    image_train_state = training_utils.TrainStateWithBatchStats.create(apply_fn = net_apply, params = init_images, tx = opt, batch_stats = None, train_it = 0, base_params = None)

    init_beta = 10 + 1
    # init_beta = 50
    # init_beta = 1

    if batch_size != -1 and batch_size < init_images['images'].shape[0]:
        print("USING BATCHED")
        
        if scale_iters:
            max_iters = int(max_iters * init_images['images'].shape[0] * batch_size)

        for i in range(max_iters + 1):
            if i% 1000 == 0:
                g_state = training_utils.init_g_state(image_train_state, init_params, final_params, beta = init_beta, use_softplus = True, img_min = 0 - train_mean, img_max = 1 - train_mean, batch_size = batch_size, both_kernels = both_kernels, batch_stats = batch_stats, has_bn = has_bn, train = False)        
            effective_i = i * batch_size/init_images['images'].shape[0]
            beta = init_beta
            # beta = min(init_beta - 1 + 10 ** (effective_i/20000), 200)
            use_softplus = False
            # use_softplus = False
            # beta = 1e6
            image_train_state, (loss, g_state, key)  = training_utils.do_training_step_recon_batched(image_train_state, train_labels, init_params, final_params, beta = beta, use_softplus = use_softplus, img_min = 0 - train_mean, img_max = 1 - train_mean, batch_size = batch_size, g_state = g_state, key = key, both_kernels = both_kernels,
                                                                                                     batch_stats = batch_stats, has_bn = has_bn, train = False, high_freq_coef = 0 if i < 10000 else 0)

            # if i == 2:
            #     sys.exit()

            if i % 10000 == 0:
                print(f'iter: {i}, loss: {loss}')
        
    else:
        print("USING BULK")
        for i in range(max_iters + 1):
            
            beta = min(init_beta - 1 + 10 ** (i/20000), 200)
            # beta = init_beta
            # use_softplus = False
            use_softplus = True
            # use_softplus = False
            # beta = 1e6
            # beta = 1.

            if pretrained:
                high_freq_coef  = 0.05 if i > 5000 else 0.00
            else:
                high_freq_coef  = 0.00

            image_train_state, (loss, acc) = training_utils.do_training_step_recon(image_train_state, train_labels, init_params, final_params, beta = beta, use_softplus = use_softplus, img_min = 0 - train_mean, img_max = 1 - train_mean, batch_stats = batch_stats, has_bn = has_bn, train = False, both_kernels = both_kernels, high_freq_coef = high_freq_coef, sparsity = sparsity)

            if i % 10000 == 0:
                print(f'iter: {i}, loss: {loss}')

    output_dict = image_train_state.params
    
    pickle.dump(output_dict, open('./{}/{}_amp_{}.pkl'.format(output_dir, save_name, amp_factor), 'wb'))

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)