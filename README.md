# Vanishing Feature: Diagnosing Model Merging and Beyond

This repository contains PyTorch implementation for results presented in the paper: *Vanishing Feature: Diagnosing Model Merging and Beyond*.

Setup the environment by running:
```bash
conda create -n vf python=3.9 cupy pkg-config libjpeg-turbo opencv  numba -c conda-forge -c pytorch && conda activate vf && conda update ffmpeg

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

The current code for preserve-first merging (PFM) and pruning are adapted from [ZipIt!](https://github.com/gstoica27/ZipIt) and [WoodFisher](https://github.com/IST-DASLab/WoodFisher), which might bring some inconvenience for further research. We plan to release a improved version in the future. For the CCA merging experiments, since the [code](https://github.com/shoroi/align-n-merge) is also adapted from ZipIt!, their implementation of the matching algorithm is directly plugged into our code in the `PFM/matching_functions.py` file.

All the dependencies in ZipIt! code are already included in the above installation, while addiontal setup needs for our code in `WoodFisher/`. Please refer `WoodFisher/README.md` for deatails.

## Structure of the repo
* `main_notebooks/` contains notebooks to reproduce most results in the main paper.
* `run_ex.py` is the main file for training models. It trains a pair of models simultaneously.
* `run_training.py` is similar to `run_ex.py`, but only trains a single model at a time.
* `training_scripts/` contains bash scripts for training models, including the hyper-parameter settings.
* `source/` contains source code for models, datasets, training, merging, etc.
  * `source/utils/opts.py` contains code for parsing arguments.
  * `source/utils/weight_matching/` contains code for weight matching.
  * `source/utils/activation_matching/` contains code for activation matching.
  * `source/utils/connect/` contains code for merging and post-merging normalization.

* `PFM/` is adapted from [ZipIt!](https://github.com/gstoica27/ZipIt), containing code to reproduce the evaluations of our prevew-first merging (PFM) framework. A complete description of the ZipIt! code can be found in the orignal repo.
  * `PFM/run_cifar.sh` is the bash script to run PFM experiments on CIFAR datasets.
  * `PFM/run_imagenet.sh` is the bash script to run PFM experiments on ImageNet dataset.
  * `PFM/run_cifar_cca.sh` is the bash script to run PFM experiments with CCA merging.
  * `PFM/visualize_results.ipynb` is the notebook to visualize the results after running the PFM experiments.
  * `PFM/calculate_params_flops.ipynb` contains the code to calculate the number of parameters and FLOPs of the models.
  * `PFM/get_zipit_premuted_models.ipynb` is used to get and save permuted models after applying ZipIt!. These models are then used for evaluating the performances of normalization methods, as shown in `main_noteooks/improve_normalization_from_vf.ipynb`.

* `WoodFisher/` is adapted from [WoodFisher](https://github.com/IST-DASLab/WoodFisher) for pruning experiemnts. A complete description of the repo can be found there.
  * `WoodFisher/main.py` is the main file to run pruning from.
  * `WoodFisher/transfer_checkpoint.ipynb` contains the code to transfer pre-trained checkpoint produced in our code to fit the WoodFisher pruning code.
  * `WoodFisher/checkpoints` is used to store the pre-trained models for the later pruning.
  * `WoodFisher/configs` contains yaml config files used for specifying training and pruning schedules. In our work, we only utilize the pruning schedules.
  * `WoodFisher/scripts` contains the all bash scripts for pruning to reproduce the results in the paper.
  * `WoodFisher/record_pruning` contains the code for visualizing the results after pruning.
  * `WoodFisher/lmc_source` contains edited code from our repo for applying re-normalization after pruning.

## Weights & Biases

We use the Weight & Biases (wandb) platform for logging results during training. To use wandb for the first time, you need to create an account and login. The `--wandb-mode` flag can be used to specify the mode of wandb. The default mode is `online`, which will log the results to the wandb server. If you want to run the code without logging to wandb, you can set `--wandb-mode` to `disabled`. If you want to log the results to wandb but do not want to create an account, you can set `--wandb-mode` to `offline`. In this case, the results will be logged to a local directory `wandb/` and you can upload the results to wandb later. For more details, please refer to the [wandb documentation](https://docs.wandb.ai/).

## Args description

### Training

We use a bash script to specify all training settings. The bash script is located in `training_scripts/`. All settings can be found in `source/utils/opts.py` with explanations. Here we only list some important args.

* `--project`: The wandb project name.
* `--run-name`: The wandb run name.
* `--dataset`: The dataset to use. We use `mnist`, `cifar10` and `cifar100` in our work.
* `--data-dir`: The dataset directory.
* `--model`: The model to use, including VGG and ResNet type of models.
  * Standard plain VGG models includ `cifar_vgg11`, `cifar_vgg13`, `cifar_vgg16`, and `cifar_vgg19`. VGG model with batch normalization is named with `_bn` suffix, e.g., `cifar_vgg11_bn`.
  * Standard ResNet models are named as `cifar_resnet[xx]`, e.g., `cifar_resnet20`. Plain/Fixup ResNet model is named with `plain_`/`fixup_` prefix, e.g., `plain_cifar_resnet20` and `fixup_cifar_resnet32`.
  * Models with layer normalization are named with `_ln` suffix, e.g., `cifar_vgg11_ln` and `cifar_resnet20_ln`.
  * Models without biases are named with `_nobias` suffix, e.g., `cifar_vgg11_nobias`.
  * Models with a larger width are named with `_[width_multipler]x` at the end, e.g., `cifar_vgg16_bn_4x`.
* `--diff-init`: Whether to use different initialization for the two models. If `True`, the two models are initialized with different random seeds.
* `--special-init`: Whether to use special initialization for models. Default is `None` and the models are initialized with the default Kaiming uniform initialization in PyTorch. If set to `vgg_init`, the Kaiming normal initialization is used.
* `--train-only`: Whether to only train the model without measuring the linear interpolation between the two models during training. If not, the linear interpolation is measured every `--lmc-freq` percent of the training.
* `--reset-bn`: Whether to reset BN statistics when measuring the linear interpolation during training.
* `--repair`: Whether to apply REPAIR/RESCALE when measuring the linear interpolation during training. Default is `None` and no re-normalizaiont is applied. If set to `repair`, REPAIR is applied. If set to `rescale`, RESCALE is applied.

### Pruning

We refer a complete description to the original repo: [WoodFisher](https://github.com/IST-DASLab/WoodFisher). Here we only list some important args in the pruning scripts located in `WoodFisher/scripts/`, which are important for reproducing the results in the paper.

* `MODULES`: The modules to prune.
* `ROOT_DIR`: The root directory to store the results.
* `DATA_DIR`: The dataset directory.
* `PRUNERS`: The pruners to use. Option: `woodfisherblock` `globalmagni` `magni` `diagfisher`
* `--num-samples`: The number of samples to use for applying re-normalization. Default is `None`.
* `--from-checkpoint-path`: The path to the pre-trained checkpoint.

## Preserve-First Merging (PFM)
The PFM framework is adapted from [ZipIt!](https://github.com/gstoica27/ZipIt), where a highly flexible implementation of model merging is provided based on graph representations of models. We modify the original repo to support our models, datasets, and the preserve-first merging framework.  The setup for the PFM repo can be found in `PFM/README.md`.

## Pruning

The pruning results reported in the paper are conducted based on the framework in [WoodFisher](https://github.com/IST-DASLab/WoodFisher). Code is stored in `WoodFisher`. We manually edit some code in the original repo to force a one-shot pruning and remove some irrelevant feautres, especially for the `WoodFisher/policies/manager.py` file, while this can also be done by modifying the pruning settings in the scripts. The original file is retained in `WoodFisher/policies/manager_ori.py`. For applying re-normalization after pruning, we merged a modified version of our code with the repo, sotred in `WoodFisher/lmc_source/`. Several lines of code are also added to `WoodFisher/policies/manager.py`. This can be used as an example to merge our code with other pruning frameworks.

We will release pre-trained checkpoints for re-producing the pruning results in the future. These checkpoints were already transferred and hence there is no need to run the `WoodFisher/transfer_checkpoint.ipynb`.

We also provide an simple and self-contained example to showcase how to apply normalization after pruning in `main_notebooks/renormalize_pruned_model.ipynb`. It uses `torch.nn.utils.prune` to prune the model and then applies normalization.

The setup for the WoodFisher repo can be found in `WoodFisher/README.md`.


## Reference

This codebase corresponds to the paper: *Vanishing Feature: Diagnosing Model Merging and Beyond*. If you use any of the code or provided models for your research, please consider citing the paper as

```bibtex
TODO
```