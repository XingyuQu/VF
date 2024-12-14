import numpy as np

config = {
    'dataset': {
        'name': 'imagenet1k',
        'train_fraction': 0.01,
        'inception_norm': False,
        # 'crop_ratio': 1.0,
        # 'res': 256,
    },
    'model': {
        'name': 'resnet50',
        'bases': [
            './checkpoints/imagenet_resnet50_1.pth', 
            './checkpoints/imagenet_resnet50_2.pth',
        ]
    },
    'eval_type': 'logits_same_task',
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
}
