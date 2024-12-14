import numpy as np

config = {
    'dataset': {
        'name': 'imagenet1k',
        'train_fraction': 0.05,
        'inception_norm': False,
        # 'crop_ratio': 1.0,
        # 'res': 256,
    },
    'model': {
        'name': 'imagenet_vgg16',
        'bases': [
            './checkpoints/imagenet_vgg16_1.pth', 
            './checkpoints/imagenet_vgg16_2.pth',
        ]
    },
    'eval_type': 'logits_same_task',
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
}
