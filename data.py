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
import tensorflow_datasets as tfds
from cub200 import CUB200

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from typing import Any, Callable, Sequence, Tuple
from flax.training import train_state, checkpoints

import matplotlib.pyplot as plt
import functools
import operator



def get_dataset(dataset_name, key, n_images):
    if dataset_name == 'mnist_odd_even':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        full_train_set = dset.MNIST(root=root, train=True, transform=None, download=True)

        all_images = np.array(full_train_set.train_data/255)
        all_labels = np.array(full_train_set.train_labels)

        even_indices = np.where(all_labels % 2 == 0)[0]
        odd_indices = np.where(all_labels % 2 == 1)[0]

        
        even_key, odd_key = jax.random.split(key)

        selected_even_indices = jax.random.choice(even_key, len(even_indices), [n_images//2], replace = False)
        selected_odd_indices = jax.random.choice(odd_key, len(odd_indices), [n_images//2], replace = False)


        train_set = jnp.zeros(shape = [n_images, 28, 28, 1])
        train_labels = (jnp.array([-1 for i in range(int(n_images//2))] + [1 for i in range(int(n_images//2))]).astype(jnp.float32)).reshape(-1, 1)
        
        

        train_set = train_set.at[:n_images//2, :, :, 0].set(jnp.take(all_images, jnp.take(even_indices, selected_even_indices), axis = 0))
        train_set = train_set.at[n_images//2:, :, :, 0].set(jnp.take(all_images, jnp.take(odd_indices, selected_odd_indices), axis = 0))
        mean = jnp.mean(train_set)

        return train_set - mean, train_labels, jnp.mean(train_set)

    if dataset_name == 'cifar10_animal_or_vehicle':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        full_train_set = dset.CIFAR10(root=root, train=True, transform=None, download=True)

        all_images = np.array(full_train_set.data/255)
        all_labels = np.array(full_train_set.targets)

        even_indices = np.where(np.isin(all_labels, [2,3,4,5,6,7]))[0]
        odd_indices = np.where(np.isin(all_labels, [0,1,8,9]))[0]

        
        even_key, odd_key = jax.random.split(key)

        selected_even_indices = jax.random.choice(even_key, len(even_indices), [n_images//2], replace = False)
        selected_odd_indices = jax.random.choice(odd_key, len(odd_indices), [n_images//2], replace = False)


        train_set = jnp.zeros(shape = [n_images, 32, 32, 3])
        train_labels = (jnp.array([-1 for i in range(int(n_images//2))] + [1 for i in range(int(n_images//2))]).astype(jnp.float32)).reshape(-1, 1)
        
        train_set = train_set.at[:n_images//2, :, :, :].set(jnp.take(all_images, jnp.take(even_indices, selected_even_indices), axis = 0))
        train_set = train_set.at[n_images//2:, :, :, :].set(jnp.take(all_images, jnp.take(odd_indices, selected_odd_indices), axis = 0))
        train_mean = jnp.mean(train_set, [0,1,2])

        return train_set - train_mean[None, None, None], train_labels, train_mean

    if dataset_name == 'mnist_all_classes':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        full_train_set = dset.MNIST(root=root, train=True, transform=None, download=True)

        all_images = np.array(full_train_set.train_data/255)
        all_labels = np.array(full_train_set.train_labels)

        key = jax.random.split(key)[0]

        im_per_class = n_images//10

        train_set = jnp.zeros(shape = [n_images, 28, 28, 1])
        train_labels = jnp.array(np.transpose(np.stack([np.eye(10) for i in range(im_per_class)], 0), [1,0,2]).reshape(-1, 10)) - 0.1

        for c in range(10):
            class_indices = np.where(all_labels % 10 == c)[0]

            selected_class_indices = jax.random.choice(key, len(class_indices), [im_per_class], replace = False)
            train_set = train_set.at[im_per_class * c: im_per_class * (c+1), :, :, 0].set(jnp.take(all_images, jnp.take(class_indices, selected_class_indices), axis = 0))

        mean = jnp.mean(train_set)

        return train_set - mean, train_labels, jnp.mean(train_set)

    if dataset_name == 'cifar10_all_classes':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        full_train_set = dset.CIFAR10(root=root, train=True, transform=None, download=True)

        all_images = np.array(full_train_set.data/255)
        all_labels = np.array(full_train_set.targets)

        key = jax.random.split(key)[0]

        im_per_class = n_images//10

        train_set = jnp.zeros(shape = [n_images, 32, 32, 3])
        train_labels = jnp.array(np.transpose(np.stack([np.eye(10) for i in range(im_per_class)], 0), [1,0,2]).reshape(-1, 10)) - 0.1

        for c in range(10):
            class_indices = np.where(all_labels % 10 == c)[0]

            selected_class_indices = jax.random.choice(key, len(class_indices), [im_per_class], replace = False)
            train_set = train_set.at[im_per_class * c: im_per_class * (c+1), :, :, :].set(jnp.take(all_images, jnp.take(class_indices, selected_class_indices), axis = 0))

        train_mean = jnp.mean(train_set, [0,1,2])

        return train_set - train_mean[None, None, None], train_labels, train_mean
    
    if dataset_name == 'cifar100_all_classes':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
            
        full_train_set = dset.CIFAR100(root=root, train=True, transform=None, download=True)

        all_images = np.array(full_train_set.data/255)
        all_labels = np.array(full_train_set.targets)

        key = jax.random.split(key)[0]

        im_per_class = n_images//100

        train_set = jnp.zeros(shape = [n_images, 32, 32, 3])
        train_labels = jnp.array(np.transpose(np.stack([np.eye(100) for i in range(im_per_class)], 0), [1,0,2]).reshape(-1, 100)) - 0.01

        for c in range(100):
            class_indices = np.where(all_labels % 100 == c)[0]

            selected_class_indices = jax.random.choice(key, len(class_indices), [im_per_class], replace = False)
            train_set = train_set.at[im_per_class * c: im_per_class * (c+1), :, :, :].set(jnp.take(all_images, jnp.take(class_indices, selected_class_indices), axis = 0))

        train_mean = jnp.mean(train_set, [0,1,2])

        return train_set - train_mean[None, None, None], train_labels, train_mean
    
    if 'cub200' in dataset_name:
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        resolution = int(dataset_name[-2:])
            
        full_train_set = CUB200(root=root, train=True, transform=None, download=True, resolution = resolution)


        # print(full_train_set._train_data[0].shape)
        # print(full_train_set._train_labels)

        all_images = np.array(full_train_set._train_data)/255
        all_labels = np.array(full_train_set._train_labels)

        key = jax.random.split(key)[0]

        n_classes = 200

        im_per_class = n_images//n_classes

        train_set = jnp.zeros(shape = [n_images, resolution, resolution, 3])
        train_labels = jnp.array(np.transpose(np.stack([np.eye(n_classes) for i in range(im_per_class)], 0), [1,0,2]).reshape(-1, n_classes)) -1/n_classes

        for c in range(n_classes):
            class_indices = np.where(all_labels % n_classes == c)[0]

            selected_class_indices = jax.random.choice(key, len(class_indices), [im_per_class], replace = False)
            train_set = train_set.at[im_per_class * c: im_per_class * (c+1), :, :, :].set(jnp.take(all_images, jnp.take(class_indices, selected_class_indices), axis = 0))

        train_mean = jnp.mean(train_set, [0,1,2])

        return train_set - train_mean[None, None, None], train_labels, train_mean
    
    
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        
class FlattenAndCast(object):
    def __init__(self, train_mean, add_dim = False):
        self.add_dim = add_dim
        self.train_mean = train_mean
        
    def __call__(self, pic):
        if self.add_dim:
            return np.array(pic, dtype=jnp.float32)[:, :, None] - self.train_mean
        
class MNISTTransform(object):
    def __init__(self, train_mean):
        self.train_mean = train_mean
        
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)[:, :, None]/255 - self.train_mean
    
class CIFARTransform(object):
    def __init__(self, train_mean):
        self.train_mean = train_mean[None, None, None]
        
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)[:, :, None]/255 - self.train_mean
        
def get_test_dataset(dataset_name, train_mean):
    if dataset_name == 'mnist_odd_even':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        
        full_test_set = dset.MNIST(root=root, train=False, transform=MNISTTransform(train_mean), download=True, target_transform = lambda x: 2 * int(x%2) - 1)
        
        loader = NumpyLoader(full_test_set, batch_size=512, shuffle=False)
        

        return loader

    if dataset_name == 'cifar10_animal_or_vehicle':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        full_test_set = dset.CIFAR10(root=root, train=False, transform=CIFARTransform(train_mean), download=True, target_transform = lambda x: 2 * int(x in [0, 1, 8, 0]) - 1)

        loader = NumpyLoader(full_test_set, batch_size=512, shuffle=False)

        return loader

    if dataset_name == 'mnist_all_classes':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        
        full_test_set = dset.MNIST(root=root, train=False, transform=MNISTTransform(train_mean), download=True, target_transform = lambda y: np.eye(10)[y] - 0.1)
        
        loader = NumpyLoader(full_test_set, batch_size=512, shuffle=False)
        

        return loader

    if dataset_name == 'cifar10_all_classes':
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        full_test_set = dset.CIFAR10(root=root, train=False, transform=CIFARTransform(train_mean), download=True, target_transform = lambda y: np.eye(10)[y] - 0.1)

        loader = NumpyLoader(full_test_set, batch_size=512, shuffle=False)

        return loader