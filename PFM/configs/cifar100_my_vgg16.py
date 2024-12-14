config = {
    'dataset': {
        'name': 'cifar100',
        'shuffle_train': True
    },
    'model': {
        'name': 'my_vgg16',
        'bases': [
            './checkpoints/cifar100_my_vgg16_1.pt',
            './checkpoints/cifar100_my_vgg16_2.pt',
        ]
    },
    'eval_type': 'logits_same_task',
    'merging_fn': 'match_tensors_permute',
    'merging_metrics': ['covariance', 'mean'],
}