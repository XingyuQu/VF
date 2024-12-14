from torchvision import transforms, datasets
import os
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from collections import defaultdict
import random
import torch


def subset_dataset(dataset, samples_per_class):
    """
    Subset a dataset to include a fixed number of samples per class.
    
    Args:
        dataset (Dataset): The dataset to subset.
        samples_per_class (int): Number of samples to keep per class.
    
    Returns:
        Subset: A subset of the original dataset with balanced classes.
    """
    class_to_indices = defaultdict(list)
    
    # Group indices by class
    for idx, (_, label) in enumerate(dataset.imgs):
        class_to_indices[label].append(idx)
    
    # Randomly sample `samples_per_class` indices for each class
    selected_indices = []
    for label, indices in class_to_indices.items():
        selected_indices.extend(random.sample(indices, min(samples_per_class, len(indices))))
    
    # Return a subset dataset
    return torch.utils.data.Subset(dataset, selected_indices)


def write(dataset, path, name):
    print(f'writing {name}...')
    writer = DatasetWriter(os.path.join(path, name), {
        'image': RGBImageField(write_mode='smart',
                               max_resolution=500,
                               compress_probability=0.5,
                               jpeg_quality=90),
        'label': IntField(),
    }, num_workers=8)

    writer.from_indexed_dataset(dataset, chunksize=100)

if __name__ == "__main__":
    data_dir = '/home/xingyu/Repos/Linear_Mode_Connectivity/data/imagenet'

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(root=traindir)
    val_dataset = datasets.ImageFolder(root=valdir)

    # Subset the training dataset to include 200 samples per class
    train_dataset = subset_dataset(train_dataset, samples_per_class=200)

    out_dir = '/home/xingyu/Repos/Linear_Mode_Connectivity/data/imagenet_ffcv/'

    write(train_dataset, out_dir, 'train.ffcv')
    write(val_dataset, out_dir, 'val.ffcv')

# def generate_random_class_splits(total_classes, split_proportions):
#     splits = []
#     start_idx = 0
#     selection_indices = np.arange(total_classes)
#     for i, split_prop in enumerate(split_proportions):
#         if i == (len(split_proportions) - 1):
#             splits += [selection_indices]
#             break

#         split_amount = int(total_classes * split_prop)
#         split_idxs, selection_indices = train_test_split(selection_indices, train_size=split_amount)
#         splits += [split_idxs]
#     return splits

# def split_even(total_classes, split_proportions):
#     splits = [[]]

#     for i in range(total_classes):
#         if len(splits[-1]) >= int(total_classes * split_proportions[0]):
#             del split_proportions[0]
#             splits.append([])
#         splits[-1].append(i)

#     return splits


# def create_subsets(train_dset, test_dset, model_class_splits):
#     class_names = test_dset.classes

#     model_loaders = []
#     for i, model_classes in enumerate(model_class_splits):
#         # Class indices
#         if isinstance(model_classes[0], int) or isinstance(model_classes[0], np.int64):
#             split_idxs = model_classes
#         # Class names
#         elif isinstance(model_classes[0], str):
#             split_idxs  = [class_names.index(i) for i in model_classes]
#         else:
#             # pdb.set_trace()
#             raise ValueError(f'unknown classes: {model_classes}')

#         train_subset_idxs = [i for i, label in enumerate(train_dset.targets) if label in split_idxs]
#         test_subset_idxs =  [i for i, label in enumerate(test_dset.targets) if label in split_idxs]

#         train_subset = torch.utils.data.Subset(train_dset, train_subset_idxs)
#         test_subset = torch.utils.data.Subset(test_dset, test_subset_idxs)

#         model_loaders.append((train_subset, test_subset))
#     return model_loaders
