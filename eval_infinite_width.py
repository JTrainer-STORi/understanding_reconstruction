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

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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

def get_ntk_batched(ntk_fn, images1, images2):    
    ntk = np.zeros(shape = [images1.shape[0], images2.shape[0]])

    batch_size = 25

    for a in range(int(np.ceil(images1.shape[0]/batch_size))):
        for b in range(int(np.ceil(images2.shape[0]/batch_size))):
            ntk[a * batch_size: (a+1) * batch_size, b * batch_size: (b+1) * batch_size] = ntk_fn(images1[a * batch_size: (a+1) * batch_size], images2[b * batch_size: (b+1) * batch_size]).ntk
            
    return ntk

def main(seed = 0, dataset_name = 'mnist_odd_even', output_dir = None, train_set_size = 100, output_name = 'eval_result', distilled_data_dir = None):
    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))
            
    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)
    test_loader = data.get_test_dataset(dataset_name, train_mean)

    if distilled_data_dir is not None:
        distilled_dict = pickle.load(open(f'./{distilled_data_dir}/distillation_result.pkl', 'rb'))
        train_images = distilled_dict['distilled_images']
        train_labels = distilled_dict['distilled_labels']
    
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Flatten(),
        stax.Dense(2048, W_std = 1, parameterization = 'ntk'),
        stax.Relu(),
        stax.Dense(2048, W_std = 1, parameterization = 'ntk'),
        stax.Relu(),
        stax.Dense(1, W_std = 1, parameterization = 'ntk')
    )

    kernel_fn = jax.jit(kernel_fn)
  
    K_ss = get_ntk_batched(kernel_fn, train_images, train_images)
    K_ss_reg = K_ss + 1e-5 * jnp.trace(K_ss) * jnp.eye(K_ss.shape[0])/K_ss.shape[0]

    solved = jnp.linalg.solve(K_ss_reg, train_labels)
    
    n_correct = 0
    n_total = 0
    
    for i, (images, labels) in enumerate(test_loader):
        print(i)
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
            
        K_ts = get_ntk_batched(kernel_fn, images, train_images)
        
        preds = K_ts @ solved

        n_correct_batch = jnp.sum((preds > 0) == (labels > 0))
        
        n_correct += n_correct_batch
        n_total += labels.shape[0]
        
    print(f'Final accuracy: {n_correct/n_total}')
    

    output_dict = {
        'n_correct': int(n_correct),
        'n_total': n_total,
        'acc': int(n_correct)/n_total,
    }

    if output_dir is not None:
        pickle.dump(output_dict, open('./{}/{}.pkl'.format(output_dir, output_name), 'wb'))

    print('done')

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)