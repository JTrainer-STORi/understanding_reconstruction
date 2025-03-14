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
from models_resnets import ResNet, Bottleneck


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

class MLP(nn.Module):
    no_bias: bool = False
    width: Any = None
    ntk_param: bool = False
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, train = True, use_softplus = False, beta = 1., return_feat = False):
        x = x.reshape((x.shape[0], -1))

        # feat_fc = x
        
        # feats.append(feat_fc)
        
        if self.ntk_param:
            x = nn.Dense(features= self.width[0], use_bias = not self.no_bias, kernel_init = nn.initializers.normal(1))(x/jnp.sqrt(x.shape[1]))
        else:
            x = nn.Dense(features= self.width[0], use_bias = not self.no_bias)(x)
        
        # from flax.linen.initializers import lecun_normal
        
        # beta = 4.
        
        if use_softplus:
            x = nn.softplus(beta * x)/beta
            # x = custom_relu(x)
        else:
            x = nn.relu(x)
        
        if self.ntk_param:
            x = nn.Dense(features= self.width[1], use_bias = not self.no_bias, kernel_init = nn.initializers.normal(1))(x/jnp.sqrt(x.shape[1]))
        else:
            x = nn.Dense(features= self.width[1], use_bias = not self.no_bias)(x)

        if use_softplus:
            x = nn.softplus(beta * x)/beta
            # x = custom_relu(x)
        else:
            x = nn.relu(x)
            
        feat = x
        
        if self.ntk_param:
            x = nn.Dense(features= self.output_dim, use_bias = not self.no_bias, kernel_init = nn.initializers.normal(1))(x/jnp.sqrt(x.shape[1]))
        else:
            x = nn.Dense(features= self.output_dim, use_bias = not self.no_bias)(x)

        
        if return_feat:
            return x, feat
        
        return x

class ResNet18(nn.Module):
    output_dim: int = 1
    use_lora: bool = False

    @nn.compact
    def __call__(self, x, train = True, use_softplus = False, beta = 1., return_feat = False):
        x = ResNet(output = 'activations', pretrained = 'imagenet', architecture = 'resnet18', normalize = False, use_lora = self.use_lora)(x, train = train, use_softplus = use_softplus, beta = beta)['fc']
        
        feat = x

        x = nn.Dense(self.output_dim)(x)

        if return_feat:
            return x, feat
        
        return x