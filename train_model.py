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

def filter_lora_params(params):
    """Returns a mask for params, where only LoRA parameters are trainable."""
    def mask_fn(param_name, _):
        return "lora_A" in param_name or "lora_B" in param_name  # Train only LoRA parameters
    return jax.tree_map_with_path(lambda path, _: mask_fn("/".join(path), _), params)

def get_noise_mask(key, shape, p):
    return (jax.random.uniform(key, shape = shape) > p).astype(jnp.float32)

def mask_noise(images, noise, noise_mask):
    return images * noise_mask + noise * (1. - noise_mask)

def main(seed = 0, dataset_name = 'mnist_odd_even', n_epochs = 1e6, lr = 1e-3, model_width = 1000, train_set_size = 100, output_dir = None, linearize = False, loss_checkpoints = [], ntk_param = False, no_bias = False, iter_checkpoints = [], distilled_data_dir = None, use_dp = False, clip_grad_norm = 2, grad_noise_ratio = 0.01, use_adam = False, second_seed = None, noise_corrupt_ratio = 0.0, pretrained = False, xent = False, momentum = 0.9, use_lora = False):
    if pretrained:
        jax.config.update("jax_enable_x64", True)

    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(locals()))

    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)

    key = jax.random.PRNGKey(seed)

    if noise_corrupt_ratio > 0.0:
        noise_key, noise_mask_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, shape = train_images.shape)
        noise_mask = get_noise_mask(noise_mask_key, train_images.shape, noise_corrupt_ratio)
        train_images = mask_noise(train_images, noise, noise_mask)
    
    if distilled_data_dir is not None:
        distilled_dict = pickle.load(open(f'./{distilled_data_dir}/distillation_result.pkl', 'rb'))
        train_images = distilled_dict['distilled_images']
        train_labels = distilled_dict['distilled_labels']

    
    print(train_labels.shape)
    if pretrained:
        model = models.ResNet18(output_dim = train_labels.shape[-1])
        has_bn = True
    else:
        model = models.MLP(width = [model_width, model_width], ntk_param = ntk_param, no_bias = no_bias, output_dim = train_labels.shape[-1])
        has_bn = False
    net_init, net_apply_base = model.init, model.apply

    if second_seed is None:
        init_params = net_init(key, train_images)['params']
        init_batch_stats = net_init(key, train_images)['batch_stats'] if has_bn else None


        if pretrained:
            init_params = to_dtype(init_params, jnp.float64)
            init_batch_stats = to_dtype(init_batch_stats, jnp.float64)
            
        # print(init_batch_stats)
    else:
        init_params = net_init(jax.random.PRNGKey(second_seed), train_images)['params']
        init_batch_stats = net_init(key, train_images)['batch_stats'] if has_bn else None

    if linearize:
        net_apply = get_linear_forward(net_apply_base, init_params, init_batch_stats, has_bn = has_bn)
    else:
        net_apply = net_apply_base
    
    if use_adam:
        opt = tx.adam(lr)
    elif not use_dp:
        opt = tx.sgd(lr, momentum = momentum)
        # opt = tx.sgd(lr, momentum = 0.0)
    else:
        opt = tx.dpsgd(lr, clip_grad_norm, grad_noise_ratio, seed, momentum = 0.9)

    if use_lora:
        trainable_mask = filter_lora_params(init_params)
        opt = tx.masked(opt, trainable_mask)

        trainable_params = jax.tree_util.tree_leaves(jax.tree_map(lambda x: x if x is not None else None, trainable_mask))
        print("Trainable Parameters:", [name for name, is_trainable in zip(init_params.keys(), trainable_params) if is_trainable])

    model_train_state = training_utils.TrainStateWithBatchStats.create(apply_fn = net_apply, params = init_params, tx = opt, batch_stats = init_batch_stats, train_it = 0, base_params = None)

    

    loss = np.inf

    alphas = np.zeros(train_labels.shape)

    for i in range(int(n_epochs)):
        model_train_state, (loss, acc, err) = training_utils.do_training_step(model_train_state, {'images': train_images, 'labels': train_labels}, use_dp = use_dp, has_bn = has_bn, train = False, xent = xent)
        alphas += np.array(err) * lr
        while len(loss_checkpoints) > 0 and loss < loss_checkpoints[0]:
            output_dict = {
                'init_params': init_params,
                'final_params': model_train_state.params,
                'final_loss': loss,
                'final_acc': acc,
                'iter': i,
            }
            print(f'saving checkpoint at loss {loss_checkpoints[0]}')
            if output_dir is not None:
                pickle.dump(output_dict, open('./{}/loss_{}_checkpoint.pkl'.format(output_dir, loss_checkpoints[0]), 'wb'))

            loss_checkpoints.pop(0)

        if len(iter_checkpoints) > 0 and i == iter_checkpoints[0]:
            output_dict = {
                'init_params': init_params,
                'final_params': model_train_state.params,
                'final_loss': loss,
                'final_acc': acc,
                'iter': i,
            }
            print(f'saving checkpoint at iter {iter_checkpoints[0]}')
            if output_dir is not None:
                pickle.dump(output_dict, open('./{}/iter_{}_checkpoint.pkl'.format(output_dir, iter_checkpoints[0]), 'wb'))

            iter_checkpoints.pop(0)
        
        if i % 10000 == 0:
            print(f'iter: {i}, loss: {loss}')
            (val, _), grad = jax.value_and_grad(training_utils.get_training_loss_l2, has_aux = True)(model_train_state.params, train_images, train_labels, model_train_state, has_bn = has_bn, batch_stats = model_train_state.batch_stats, xent = xent)


        if loss < 1e-10:
            print("Loss is really small, exiting early")
            break
    
    print(f'iter: {i}, loss: {loss}')

    output_dict = {
        'init_params': init_params,
        'final_params': model_train_state.params,
        'final_loss': loss,
        'iter': i,
        'alphas': alphas,
        'batch_stats': model_train_state.batch_stats,
    }

    training_dict = {
        'train_images': train_images,
        'train_labels': train_labels,
        'train_mean': train_mean
    }

    print('done')

    if output_dir is not None:
        pickle.dump(output_dict, open('./{}/final_checkpoint.pkl'.format(output_dir), 'wb'))
        pickle.dump(training_dict, open('./{}/training_set.pkl'.format(output_dir), 'wb'))

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)