import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATION'] = "false"

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
import make_reconstruction

def get_ntk(apply, params, train_images):
    def combined_apply(params, x):
        return apply({'params': params}, x, mutable = [])[0]
    
    emp_kernel_fn = jax.jit(nt.empirical_kernel_fn(f = combined_apply))
    
    ntk = np.zeros(shape = [train_images.shape[0], train_images.shape[0]])

    batch_size = 25

    for a in range(int(np.ceil(train_images.shape[0]/batch_size))):
        for b in range(int(np.ceil(train_images.shape[0]/batch_size))):
            ntk[a * batch_size: (a+1) * batch_size, b * batch_size: (b+1) * batch_size] = emp_kernel_fn(train_images[a * batch_size: (a+1) * batch_size], train_images[b * batch_size: (b+1) * batch_size], None, params).ntk
            
    return ntk

def main(seed = 0, dataset_name = 'mnist_odd_even', saved_network_dir = None, linearize = False, ntk_param = False, no_bias = False, output_dir = None, model_width = 4096, train_set_size = 100, n_per_class_distilled = 1):
    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)

    key = jax.random.PRNGKey(seed)

    amp = (2 * n_per_class_distilled)/train_images.shape[0]

    make_reconstruction.main(seed = seed, output_dir = saved_network_dir, dataset_name = dataset_name, amp_factor = amp, model_width = model_width, train_set_size = train_set_size, save_name = 'distill_from_trained_output', linearize = linearize)

    recon_dict = pickle.load(open(f'./{saved_network_dir}/distill_from_trained_output_amp_{amp}.pkl', 'rb'))
    checkpoint_dict = pickle.load(open(f'./{saved_network_dir}/final_checkpoint.pkl', 'rb'))

    init_params = checkpoint_dict['init_params']
    final_params = checkpoint_dict['final_params']

    recon_images = recon_dict['images']
    duals = recon_dict['duals']

    model = models.MLP(width = [model_width, model_width], ntk_param = ntk_param, no_bias = no_bias)
    net_init, net_apply_base = model.init, model.apply

    if linearize:
        net_apply = get_linear_forward(model.apply, init_params)
    else:
        net_apply = model.apply

    K = get_ntk(net_apply, final_params, recon_images)
    
    labels = K @ duals.reshape(-1, 1)


    output_dict = {
        'distilled_images': recon_images,
        'distilled_labels': labels,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_mean': train_mean
    }


    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))
        pickle.dump(output_dict, open('./{}/distillation_result.pkl'.format(output_dir), 'wb'))

    print('done')

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)