"""Calculate mean and std."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

from torchvision import datasets, transforms

train_transform = transforms.Compose([transforms.ToTensor()])

# cifar10
train_set = datasets.CIFAR10(root='~/DATASET/PyTorch-Dataset/', train=True, download=True, transform=train_transform)
print("CIFAR10 train data shape:", train_set.data.shape)
print("CIFAR10 train data mean: ", train_set.data.mean(axis=(0, 1, 2))/255)
print("CIFAR10 train data std:  ", train_set.data.std(axis=(0, 1, 2))/255)
# (50000, 32, 32, 3)
# [0.49139968  0.48215841  0.44653091]
# [0.24703223  0.24348513  0.26158784]


# cifar100
train_set = datasets.CIFAR100(root='~/DATASET/PyTorch-Dataset/', train=True, download=True, transform=train_transform)
print("CIFAR100 train data shape:", train_set.data.shape)
print("CIFAR100 train data mean: ", train_set.data.mean(axis=(0, 1, 2, 3))/255)
print("CIFAR100 train data std:  ", train_set.data.std(axis=(0, 1, 2, 3))/255)
# (50000, 32, 32, 3)
#(0.4782)
#(0.2682)

# mnist
train_set = datasets.MNIST(root='~/DATASET/PyTorch-Dataset/', train=True, download=True, transform=train_transform)
print("MNIST train data shape:", train_set.data.shape)
print("MNIST train data mean: ", train_set.data.float().mean()/255)
print("MNIST train data std:  ", train_set.data.float().std()/255)
