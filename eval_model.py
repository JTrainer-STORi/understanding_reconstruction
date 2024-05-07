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
import utils
import models
import training_utils
import pickle

def main(seed = 0, dataset_name = 'mnist_odd_even', output_dir = None, train_set_size = 100, model_width = 1000, ntk_param = False, no_bias = False, checkpoint_name = 'final_checkpoint', output_name = 'eval_result', linearize = False):
    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(locals()))

    train_images, train_labels, train_mean = data.get_dataset(dataset_name, jax.random.PRNGKey(seed), train_set_size)
    test_loader = data.get_test_dataset(dataset_name, train_mean)
    
    checkpoint_dict = pickle.load(open('./{}/{}.pkl'.format(output_dir, checkpoint_name), 'rb'))
    
    init_params = checkpoint_dict['init_params']
    final_params = checkpoint_dict['final_params']

    model = models.MLP(width = [model_width, model_width], ntk_param = ntk_param, no_bias = no_bias, output_dim = train_labels.shape[-1])
    
    if linearize:
        net_apply = get_linear_forward(model.apply, init_params)
    else:
        net_apply = model.apply

    
    
    model_train_state = training_utils.TrainStateWithBatchStats.create(apply_fn = net_apply, params = final_params, tx = tx.sgd(0.0), batch_stats = None, train_it = 0, base_params = None)
    
    
    n_correct = 0
    n_total = 0
    
    for i, (images, labels) in enumerate(test_loader):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
            
        _, (_, _, n_correct_batch, _)= training_utils.get_training_loss_l2(model_train_state.params, images, labels, model_train_state, l2 = 0., train = False, has_bn = False, batch_stats = None, use_base_params = False)
        
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

    return output_dict

if __name__ == '__main__':
    # main('cifar10')
    fire.Fire(main)