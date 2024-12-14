import os
import torch
from lmc_utils import BatchScale1d, BatchScale2d, interpolate_state_dicts
from copy import deepcopy
from model_merger import ModelMerge
from graphs.base_graph import NodeType
from torch import nn
from utils import get_config_from_name, get_device, prepare_experiment_config, get_merging_fn

from lmc_utils import ResetLayer, RescaleLayer, TrackLayer

def make_repaired_imagenet_vgg16(net, device=None):
    net1 = deepcopy(net).to(device)
    for i, layer in enumerate(net1.features):
        if isinstance(layer, (nn.Conv2d)):
            net1.features[i] = ResetLayer(layer)
    for i, layer in enumerate(net1.classifier):
        if i < 4 and isinstance(layer, nn.Linear):
            net1.classifier[i] = ResetLayer(layer)
    return net1.eval().to(device)

def make_rescaled_imagenet_vgg16(net, device=None):
    net1 = deepcopy(net).to(device)
    for i, layer in enumerate(net1.features):
        if isinstance(layer, (nn.Conv2d)):
            net1.features[i] = RescaleLayer(layer)
    for i, layer in enumerate(net1.classifier):
        if i < 4 and isinstance(layer, nn.Linear):
            net1.classifier[i] = RescaleLayer(layer)
    return net1.eval().to(device)


def make_tracked_imagenet_vgg16(net, device=None):
    net1 = deepcopy(net)
    for i, layer in enumerate(net1.features):
        if isinstance(layer, (nn.Conv2d)):
            net1.features[i] = TrackLayer(layer)
    for i, layer in enumerate(net1.classifier):
        if i < 4 and isinstance(layer, nn.Linear):
            net1.classifier[i] = TrackLayer(layer)
    return net1.eval().to(device)


def validate(model, testloader, criterion, device, half=False, num_iters=None, print_freq=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        it = 0
        for data in testloader:
            if num_iters is not None and it >= num_iters:
                break
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            it += 1
            if print_freq is not None and it % print_freq == 0:
                print('Accuracy so far: {}%'.format(100 * correct / total))
        
    print('Accuracy of the network on the 10000 test images: {}%'.format(100 * correct / total))
    return loss_sum / total, correct / total


def validate_ensemble(models, testloader, criterion, device, half=False, num_iters=None, print_freq=None):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    
    with torch.no_grad():
        it = 0
        for data in testloader:
            if num_iters is not None and it >= num_iters:
                break
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device).long()
            outputs = sum(model(images) for model in models)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            it += 1
            if print_freq is not None and it % print_freq == 0:
                print('Accuracy so far: {}%'.format(100 * correct / total))
    
    print('Accuracy of the network on the 10000 test images: {}%'.format(100 * correct / total))
    return loss_sum / total, correct / total


def imagenet_reset_bn_stats(model, loader, reset=True, num_iters=None):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) in (nn.BatchNorm2d, BatchScale2d, BatchScale1d, nn.BatchNorm1d):
            if reset:
                m.momentum = None # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    # run a single train epoch with augmentations to recalc stats
    model.train()
    iter = 0
    with torch.no_grad():
        print('Resetting batch norm stats')
        for images, _ in loader:
            if num_iters is not None and iter >= num_iters:
                break
            if iter == len(loader): # hack for fractional loader
                break
            images = images.to(device).float()
            _ = model(images)
            iter += 1
    model.eval()
    return model


# python imagenet_experiments.py --device cuda:0 --config imagenet_vgg16 --save_dir pfm_results/imagenet --pair 1_2 --test
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config', type=str, default='imagenet_vgg16', choices=['imagenet_vgg16', 'imagenet_resnet50'])
    parser.add_argument('--save_dir', type=str, default='pfm_results/imagenet/')
    parser.add_argument('--pair', type=str, default='1_2')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    save_prefix = args.save_dir + args.config + '_' + args.pair + '_results'
    if os.path.exists(save_prefix + '.pth'):
        print(f"Results already exist for {args.config} pair {args.pair} at {save_prefix + '.pth'}")
        return
    
    config_name = args.config

    device = args.device
    raw_config = get_config_from_name(config_name, device=device)

    # change the model bases to the desired pair
    pairs = args.pair.split('_')
    pairs = [int(pair) for pair in pairs]
    for i, model_idx in enumerate(pairs):
        path = raw_config['model']['bases'][i]  # ..._1.pth
        # replace the last digit with the model_idx
        path = path[:-5] + str(model_idx) + '.pth'
        # remove ./ from the path
        # if path.startswith('./'):
        #     path = path[2:]
        print(path)
        raw_config['model']['bases'][i] = path
    
    model_paths = deepcopy(raw_config['model']['bases'])

    cur_config = deepcopy(raw_config)
    config = prepare_experiment_config(cur_config)

    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']
    train_loader.batch_size = 32
    test_loader.batch_size = 32
    test_loader.num_workers = 0
    print(f"Training samples: {train_loader.batch_size * len(train_loader)}")
    print(f"Testing samples: {test_loader.batch_size * len(test_loader)}")

    # use dummy data for testing; create 10 dummy_input by TensorDataset for each loader; dataset is cifar10 and cifar100
    if args.test:
        dummy_data = torch.randn(10, 3, 32, 32)
        class_num = 1000
        dummy_label = torch.randint(0, class_num, (10,))
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data, dummy_label), batch_size=10)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data, dummy_label), batch_size=10)

    base_models = [base_model for base_model in config['models']['bases']]
    Grapher = config['graph']

    criterion = torch.nn.CrossEntropyLoss()

    if 'resnet' in config_name:
        merging_fn_s = ['match_tensors_permute', 'match_tensors_identity']
    else:
        merging_fn_s = ['match_tensors_permute']
    if args.test:
        merging_fn_s = ['match_tensors_permute']

    # run one pass to record metrics for merging_fn
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model
                      in base_models]
    Merge = ModelMerge(*graphs, device=device)
    Merge.transform(
        deepcopy(config['models']['new']),
        train_loader,
        transform_fn=get_merging_fn("match_tensors_permute"),
        metric_classes=config['metric_fns'],
        stop_at=None,
        start_at=None
    )
    metrics = Merge.metrics
    metrics_save_path = 'pfm_results/imagenet/imagenet_resnet50_1_2_metrics.pth'
    torch.save(metrics, metrics_save_path)

    res_dict = {merging_fn: {'bias_corr': {'loss': [], 'acc': []},
                             'rescale': {'loss': [], 'acc': []}, 'repair': {'loss': [], 'acc': []},
                             'merger': {'loss': [], 'acc': []}, 'reset': {'loss': [], 'acc': []}} for merging_fn in merging_fn_s}
    res_dict.update({'models': model_paths})
    if 'vgg' in config_name and 'bn' not in config_name:
        res_dict.update({'ensemble': {'loss': [], 'acc': []}})
        res_dict.update({'base_0': {'loss': [], 'acc': []}})
        res_dict.update({'base_1': {'loss': [], 'acc': []}})

    # validate base models
    for i, base_model in enumerate(base_models):
        print(f"base_model_{i}")
        base_loss, base_acc = validate(base_model, test_loader, criterion, device)
        print(f"base_loss_{i}: {base_loss}", f"base_acc_{i}: {base_acc}")
        res_dict.update({f'base_{i}': {'loss': base_loss, 'acc': base_acc}})

    # validate ensemble
    print("Testing ensemble")
    ensemble_loss, ensemble_acc = validate_ensemble(base_models, test_loader, criterion, device)
    print(f"ensemble_loss: {ensemble_loss}", f"ensemble_acc: {ensemble_acc}")
    res_dict.update({'ensemble': {'loss': ensemble_loss, 'acc': ensemble_acc}})
    
    
    # get prefix_nodes
    prefix_nodes = []
    for node in graphs[0].G.nodes:
        node_info = graphs[0].get_node_info(node)
        if node_info['type'] == NodeType.PREFIX:
            prefix_nodes.append(node)
    prefix_nodes = [None] + prefix_nodes
    # prefix_nodes = [50]

    for merging_fn in merging_fn_s:
        print(f"merging_fn: {merging_fn}")
        stop_at = None
        for start_at in prefix_nodes:
            print(f"start_at: {start_at}")
            graphs = [Grapher(deepcopy(base_model)).graphify() for base_model
                      in base_models]
            Merge = ModelMerge(*graphs, device=device)
            prepared_metrics = torch.load(metrics_save_path) # metrics

            Merge.transform(
                deepcopy(config['models']['new']), 
                train_loader, 
                transform_fn=get_merging_fn(merging_fn),
                metric_classes=config['metric_fns'],
                stop_at=stop_at,
                start_at=start_at,
                prepared_metrics=prepared_metrics,
                # a=0.3,
                # b=0.8
            )
            merged_model_backup = deepcopy(Merge.merged_model)  # hack for hooks

            merger_loss, merger_acc = validate(Merge, test_loader, criterion, device, print_freq=500)
            print(f"merger_loss: {merger_loss}", f"merger_acc: {merger_acc}")
            res_dict[merging_fn]['merger']['loss'].append(merger_loss)
            res_dict[merging_fn]['merger']['acc'].append(merger_acc)


            # reset
            if 'bn' in config_name or ('resnet' in config_name):
                imagenet_reset_bn_stats(Merge, train_loader)
                reset_loss, reset_acc = validate(Merge, test_loader, criterion, device, print_freq=500)
                print(f"reset_loss: {reset_loss}", f"reset_acc: {reset_acc}")
                res_dict[merging_fn]['reset']['loss'].append(reset_loss)
                res_dict[merging_fn]['reset']['acc'].append(reset_acc)
            else:

                print("---Permuting the models---")
                graphs = Merge.graphs
                base_model_merge_s = [deepcopy(graph.model) for graph in graphs]
                # remove all hooks from the model
                for model in base_model_merge_s:
                    model._forward_hooks = {}
                    model._backward_hooks = {}
                
                ### bias correction ###
                Merge.merged_model = deepcopy(merged_model_backup)
                merged_sd = Merge.merged_model.state_dict()
                merged_sd = interpolate_state_dicts(merged_sd, merged_sd, 0.5, True)
                Merge.merged_model.load_state_dict(deepcopy(merged_sd))
                bias_corr_loss, bias_corr_acc = validate(Merge, test_loader, criterion, device, print_freq=500)
                print(f"bias_corr_loss: {bias_corr_loss}", f"bias_corr_acc: {bias_corr_acc}")
                res_dict[merging_fn]['bias_corr']['loss'].append(bias_corr_loss)
                res_dict[merging_fn]['bias_corr']['acc'].append(bias_corr_acc)

                # prepare tracked models for repair and rescale
                model_tracked_s = [make_tracked_imagenet_vgg16(model, device) for model in base_model_merge_s]
                for model in model_tracked_s:
                    imagenet_reset_bn_stats(model, train_loader)
                num_models = len(model_tracked_s)
                alpha_s = [1/num_models] * num_models

                means = [[], []]
                stds = [[], []]
                goal_means = []
                goal_stds = []
                for layers in zip(*[model_tracked.modules() for model_tracked in model_tracked_s]):
                    
                    if not isinstance(layers[0], TrackLayer):
                        continue
                    # get neuronal statistics of original networks
                    mu_s = [layer.get_stats()[0] for layer in layers[:-1]]
                    std_s = [layer.get_stats()[1] for layer in layers[:-1]]
                    
                    goal_mean = sum([alpha * mu for alpha, mu in zip(alpha_s, mu_s)])
                    goal_std = sum([alpha * std for alpha, std in zip(alpha_s, std_s)])
                    
                    means[0].append(mu_s)
                    stds[0].append(std_s)
                    goal_means.append(goal_mean)
                    goal_stds.append(goal_std)


                # repair
                Merge.merged_model = deepcopy(merged_model_backup)
                model_repaired = Merge.merged_model
                variant = 'repair'

                if variant == 'repair':
                    model_repaired = make_repaired_imagenet_vgg16(model_repaired, device)
                elif variant == 'rescale':
                    model_repaired = make_rescaled_imagenet_vgg16(model_repaired, device)

                i = 0
                for layer in model_repaired.modules():
                    if isinstance(layer, (ResetLayer)):
                        layer.set_stats(goal_means[i], goal_stds[i])
                        i += 1
                    elif isinstance(layer, (RescaleLayer)):
                        layer.set_stats(goal_stds[i])
                        i += 1

                Merge.merged_model = model_repaired
                imagenet_reset_bn_stats(Merge, train_loader)
                repair_loss, repair_acc = validate(Merge, test_loader, criterion, device, print_freq=500)
                print(f"repair_loss: {repair_loss}", f"repair_acc: {repair_acc}")
                res_dict[merging_fn]['repair']['loss'].append(repair_loss)
                res_dict[merging_fn]['repair']['acc'].append(repair_acc)

                # rescale
                Merge.merged_model = deepcopy(merged_model_backup)
                model_repaired = Merge.merged_model
                variant = 'rescale'

                if variant == 'repair':
                    model_repaired = make_repaired_imagenet_vgg16(model_repaired, device)
                elif variant == 'rescale':
                    model_repaired = make_rescaled_imagenet_vgg16(model_repaired, device)

                i = 0
                for layer in model_repaired.modules():
                    if isinstance(layer, (ResetLayer)):
                        layer.set_stats(goal_means[i], goal_stds[i])
                        i += 1
                    elif isinstance(layer, (RescaleLayer)):
                        layer.set_stats(goal_stds[i])
                        i += 1

                Merge.merged_model = model_repaired
                imagenet_reset_bn_stats(Merge, train_loader)
                rescale_loss, rescale_acc = validate(Merge, test_loader, criterion, device, print_freq=500)
                print(f"rescale_loss: {rescale_loss}", f"rescale_acc: {rescale_acc}")
                res_dict[merging_fn]['rescale']['loss'].append(rescale_loss)
                res_dict[merging_fn]['rescale']['acc'].append(rescale_acc)


    # convert list to ndarray
    for merging_fn in merging_fn_s:
        for key in res_dict[merging_fn].keys():
            for metric in ['loss', 'acc']:
                res_dict[merging_fn][key][metric] = torch.tensor(res_dict[merging_fn][key][metric])
    ## save results in a file ##
    save_prefix = args.save_dir + args.config + '_' + args.pair + '_results'
    torch.save(res_dict, save_prefix + '.pth')
    print(f"Results saved at {save_prefix}.pth")


if __name__ == '__main__':
    main()
