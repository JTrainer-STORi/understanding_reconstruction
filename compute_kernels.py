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

def get_ntk(apply, params, train_images):
    def combined_apply(params, x):
        return apply({'params': params}, x, mutable = [])[0]
    
    emp_kernel_fn = jax.jit(nt.empirical_kernel_fn(f = combined_apply))
    
    ntk = np.zeros(shape = [train_images.shape[0], train_images.shape[0]])

    batch_size = 5

    for a in range(int(np.ceil(train_images.shape[0]/batch_size))):
        for b in range(int(np.ceil(train_images.shape[0]/batch_size))):
            ntk[a * batch_size: (a+1) * batch_size, b * batch_size: (b+1) * batch_size] = emp_kernel_fn(train_images[a * batch_size: (a+1) * batch_size], train_images[b * batch_size: (b+1) * batch_size], None, params).ntk
            
    return ntk
            

def main(seed = 0, dataset_name = 'mnist_odd_even', output_dir = None, train_set_size = 100, model_width = 1000, ntk_param = False, no_bias = False, checkpoint_name = 'final_checkpoint', linearize = False):
    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(locals()))

    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)
    
    checkpoint_dict = pickle.load(open('./{}/{}.pkl'.format(output_dir, checkpoint_name), 'rb'))
    
    init_params = checkpoint_dict['init_params']
    final_params = checkpoint_dict['final_params']
    
    
    model = models.MLP(width = [model_width, model_width], ntk_param = ntk_param, no_bias = no_bias, output_dim = train_labels.shape[-1])
    
    if linearize:
        net_apply = get_linear_forward(model.apply, init_params)
    else:
        net_apply = model.apply
    
    print("Computing init kernel")
    init_ntk = get_ntk(net_apply, init_params, train_images)

    if not linearize:
        print("Computing final kernel")
        final_ntk = get_ntk(net_apply, final_params, train_images)
    else:
        print("Computing final kernel")
        final_ntk = init_ntk
        
    output_dict = {
        'init_ntk': init_ntk,
        'final_ntk': final_ntk,
    }


    if output_dir is not None:
        pickle.dump(output_dict, open('./{}/{}_kernels.pkl'.format(output_dir, checkpoint_name), 'wb'))

    print('done')

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)