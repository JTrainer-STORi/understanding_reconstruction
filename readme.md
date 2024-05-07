# Understanding Reconstruction Attacks with the Neural Tangent Kernel and Dataset Distillation

This is the code for the ICLR 2024 paper [Understanding Reconstruction Attacks with the Neural Tangent Kernel and Dataset Distillation](https://openreview.net/forum?id=VoLDkQ6yR3)

![Attacking Fine Tuned Models](https://github.com/yolky/understanding_reconstruction/blob/main/fine_tune.png?raw=true)


# Abstract
Modern deep learning requires large volumes of data, which could contain sensitive or private information that cannot be leaked. Recent work has shown for homogeneous neural networks a large portion of this training data could be reconstructed with only access to the trained network parameters. While the attack was shown to work empirically, there exists little formal understanding of its effective regime and which datapoints are susceptible to reconstruction. In this work, we first build a stronger version of the dataset reconstruction attack and show how it can provably recover the **entire training set** in the infinite width regime. We then empirically study the characteristics of this attack on two-layer networks and reveal that its success heavily depends on deviations from the frozen infinite-width Neural Tangent Kernel limit. Next, we study the nature of easily-reconstructed images. We show that both theoretically and empirically, reconstructed images tend to "outliers" in the dataset, and that these reconstruction attacks can be used for *dataset distillation*, that is, we can retrain on reconstructed images and obtain high predictive accuracy.

# File Overview

Here is a brief overview of what each of the files does:

`train_model.py` - For training finite-width networks on either MNIST Odd/Even, CIFAR-10 Animal/Vehicle, MNIST 10 way, CIFAR-10 10 way, using either real or distilled data

`make_reconstruction.py` - For making reconstruction from the trained networks made using `train_model.py`.

`eval_infinite_width.py` - For get infinite-width accuracies of either distilled or original datasets

`distill_dataset.py` - For running KIP or RKIP distillation

`distill_from_trained.py` - For running RKIP-finite

`eval_model.py` - Evaluating test accuracy of models

Everything else is a utility file.

# Reconstructing MNIST example

We have visualizations of the attack on MNIST and CIFAR-10 in `usage_example.ipynb`. We outline basic usage, attacking a 2048-width model trained on binary MNIST
```
#training a 2048 width model on binary mnist
python3 train_model.py --n_epochs 1e6 --output_dir ./example_mnist/ --lr 4e-5 --dataset_name mnist_odd_even --train_set_size 200 --model_width 2048

#Attack the saved model
python3 make_reconstruction.py --output_dir ./example_mnist/ --dataset_name mnist_odd_even --train_set_size 200 --model_width 2048
```

To used linearized models, use the `--linearize` flag in all files.

# Attacking Fine-tuned models

See the example usage in `fine_tune_example.ipynb`

# Distillation

To use RKIP distillation run the following:

```
python3 distill_dataset.py --n_per_class_distilled 10 --train_set_size 500 --output_dir ./example_distilled_rkip --use_mse --dataset_name mnist_odd_even --rkip
```
To use KIP, omit the `--rkip` flag

To evaluate the distilled set on finite models:

```
#training
python3 train_model.py --n_epochs 1e5 --output_dir ./example_distilled_rkip/trained_model/ --lr 1e-5 --dataset_name mnist_odd_even --train_set_size 200 --model_width 4096 --distilled_data_dir ./example_distilled_rkip

#evaluation
python3 eval_model.py --output_dir ./example_distilled_rkip/trained_model/ ---dataset_name mnist_odd_even --model_width 4096
```

# Training on reconstructed images (finite-RKIP)

Example usage:
```
#training the finite model
python3 train_model.py --n_epochs 1e6 --output_dir ./example_mnist_4096/ --lr 4e-5 --dataset_name mnist_odd_even --train_set_size 200 --model_width 4096

#distilling from the finite model
python3 distill_from_trained.py --saved_network_dir ./example_mnist_4096/ --n_per_class_distilled 10 --train_set_size 200 --output_dir ./example_distilled_finite_rkip --train_set_size 200 --model_width 4096

#training from the distilled dataset
python3 train_model.py --n_epochs 1e5 --output_dir ./example_distilled_finite_rkip/trained_model/ --lr 1e-5 --dataset_name mnist_odd_even --train_set_size 200 --model_width 4096 --distilled_data_dir ./example_distilled_finite_rkip

#evaluating
python3 eval_model.py --output_dir ./example_distilled_finite_rkip/trained_model/ ---dataset_name mnist_odd_even --model_width 4096
```

# Computing Neural Tangent Kernels
Example command:
```
python3 compute_kernels.py --output_dir ./example_mnist/ --dataset_name mnist_odd_even --train_set_size 200 --model_width 2048
```
