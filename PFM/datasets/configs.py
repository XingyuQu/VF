import torchvision
from .imagenet import ImageNet1k

cifar10 = {
    'dir': '../data/',
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 128,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 0,
}

cifar100 = {
    'dir': '../data/',
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 128,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 0,
}

imagenet1k = {
    'dir': '../data/imagenet_ffcv/',
    'num_classes': 1000,
    'wrapper': ImageNet1k,
    'batch_size': 32,
    'res': 224,
    'inception_norm': True,
    'shuffle_test': False,
    'type': 'imagenet',
    'num_workers': 0,
}