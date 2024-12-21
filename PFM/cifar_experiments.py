import torch
from copy import deepcopy
import os
from graphs.base_graph import NodeType
from utils import get_config_from_name, prepare_experiment_config,\
     get_merging_fn
from lmc_utils import reset_bn_stats
from model_merger import ModelMerge
from lmc_utils import interpolate_state_dicts, repair


def validate(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: \
        {}'.format(100 * correct / total))
    return loss_sum / total, correct / total


def validate_ensemble(models, testloader, criterion, device):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = None
            for model in models:
                if outputs is None:
                    outputs = model(images) / len(models)
                else:
                    outputs += model(images) / len(models)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: \
        {}'.format(100 * correct / total))
    return loss_sum / total, correct / total


# python cifar_experiments.py --device cuda:0 --config cifar10_my_vgg16_bn --save_dir pfm_results/ --pair 1_2 --test
def main():
    from tqdm import tqdm
    tqdm.disable = True
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config', type=str, default='cifar10_my_vgg16', choices=['cifar10_my_vgg16', 'cifar10_my_vgg16_bn',
                                                                                   'cifar10_my_resnet20', 'cifar10_my_plain_resnet20',
                                                                                   'cifar10_my_resnet20_4x',
                                                                                   'cifar100_my_vgg16', 'cifar100_my_vgg16_bn',
                                                                                   'cifar100_my_resnet20_4x',])
    parser.add_argument('--save_dir', type=str, default='pfm_results/')
    parser.add_argument('--pair', type=str, default='1_2')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    print("New experiment")
    os.makedirs(args.save_dir, exist_ok=True)
    save_prefix = args.save_dir + args.config + '_' + args.pair + '_results'
    if os.path.exists(save_prefix + '.pth'):
        print(f"Results already exist for {args.config} pair {args.pair}")
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
        path = path[:-5] + '_' + str(model_idx) + '.pt'
        # remove ./ from the path
        # if path.startswith('./'):
        #     path = path[2:]
        raw_config['model']['bases'][i] = path

    model_paths = deepcopy(raw_config['model']['bases'])
    cur_config = deepcopy(raw_config)
    config = prepare_experiment_config(cur_config)

    train_loader = config['data']['train']['full']
    test_loader = config['data']['test']['full']

    # use dummy data for testing; create 10 dummy_input by TensorDataset for each loader; dataset is cifar10 and cifar100
    if args.test:
        dummy_data = torch.randn(10, 3, 32, 32)
        class_num = 10 if 'cifar10' in config_name else 100
        dummy_label = torch.randint(0, class_num, (10,))
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data, dummy_label), batch_size=10)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data, dummy_label), batch_size=10)

    base_models = [base_model for base_model in
                   config['models']['bases']]
    Grapher = config['graph']

    criterion = torch.nn.CrossEntropyLoss()

    merging_fn_s = ['match_tensors_permute', 'match_tensors_zipit',
                    'match_tensors_identity']
    if args.test:
        merging_fn_s = ['match_tensors_permute']

    res_dict = {merging_fn: {'bias_corr': {'loss': [], 'acc': []}, 'zero_bias': {'loss': [], 'acc': []},
                             'rescale': {'loss': [], 'acc': []}, 'repair': {'loss': [], 'acc': []},
                             'merger': {'loss': [], 'acc': []}, 'reset': {'loss': [], 'acc': []}} for merging_fn in merging_fn_s}
    res_dict.update({'models': model_paths})

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

    # run one pass to record metrics for merging_fn
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model
                      in base_models]
    Merge = ModelMerge(*graphs, device=device)
    Merge.transform(
        deepcopy(config['models']['new']),
        train_loader,
        transform_fn=get_merging_fn("match_tensors_identity"),
        metric_classes=config['metric_fns'],
        stop_at=None,
        start_at=None
    )
    metrics = Merge.metrics
    metrics_save_path = args.save_dir + args.config + '_' + args.pair + '_metrics.pth'
    torch.save(metrics, metrics_save_path)
    
    # get prefix_nodes
    prefix_nodes = []
    for node in graphs[0].G.nodes:
        node_info = graphs[0].get_node_info(node)
        if node_info['type'] == NodeType.PREFIX:
            prefix_nodes.append(node)
    prefix_nodes = [None] + prefix_nodes

    for merging_fn in merging_fn_s:
        print(f"merging_fn: {merging_fn}")
        stop_at = None
        for start_at in prefix_nodes:
            print(f"start_at: {start_at}")
            graphs = [Grapher(deepcopy(base_model)).graphify() for base_model
                      in base_models]
            Merge = ModelMerge(*graphs, device=device)
            Merge.transform(
                deepcopy(config['models']['new']),
                train_loader,
                transform_fn=get_merging_fn(merging_fn),
                metric_classes=config['metric_fns'],
                stop_at=stop_at,
                start_at=start_at,
                prepared_metrics=metrics
            )
            merged_model_backup = deepcopy(Merge.merged_model)  # hack for hooks

            merger_loss, merger_acc = validate(Merge, test_loader, criterion, device)
            print(f"merger_loss: {merger_loss}", f"merger_acc: {merger_acc}")
            res_dict[merging_fn]['merger']['loss'].append(merger_loss)
            res_dict[merging_fn]['merger']['acc'].append(merger_acc)

            # reset
            if 'bn' in config_name or ('plain' not in config_name and 'resnet' in config_name):
                reset_bn_stats(Merge, train_loader)
                reset_loss, reset_acc = validate(Merge, test_loader, criterion, device)
                print(f"reset_loss: {reset_loss}", f"reset_acc: {reset_acc}")
                res_dict[merging_fn]['reset']['loss'].append(reset_loss)
                res_dict[merging_fn]['reset']['acc'].append(reset_acc)
            else:
                # permute the models 
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
                Merge.merged_model.load_state_dict(merged_sd)
                bias_corr_loss, bias_corr_acc = validate(Merge, test_loader, criterion, device)
                print(f"bias_corr_loss: {bias_corr_loss}", f"bias_corr_acc: {bias_corr_acc}")
                res_dict[merging_fn]['bias_corr']['loss'].append(bias_corr_loss)
                res_dict[merging_fn]['bias_corr']['acc'].append(bias_corr_acc)

                ### zero bias ###
                Merge.merged_model = deepcopy(merged_model_backup)
                merged_sd = Merge.merged_model.state_dict()
                for key in merged_sd.keys():
                    if 'bias' in key:
                        merged_sd[key] = torch.zeros_like(merged_sd[key])
                Merge.merged_model.load_state_dict(merged_sd)
                zero_bias_loss, zero_bias_acc = validate(Merge, test_loader, criterion, device)
                print(f"zero_bias_loss: {zero_bias_loss}", f"zero_bias_acc: {zero_bias_acc}")
                res_dict[merging_fn]['zero_bias']['loss'].append(zero_bias_loss)
                res_dict[merging_fn]['zero_bias']['acc'].append(zero_bias_acc)

                # repair
                Merge.merged_model = deepcopy(merged_model_backup)
                repaired_merged_model = repair(train_loader, base_model_merge_s,
                                            Merge.merged_model, device,
                                            name=config['model']['name'], variant='repair',
                                            reset_bn=False) # reset bn in merger
                Merge.merged_model = repaired_merged_model
                reset_bn_stats(Merge, train_loader)
                repair_loss, repair_acc = validate(Merge, test_loader, criterion, device)
                print(f"repair_loss: {repair_loss}", f"repair_acc: {repair_acc}")
                res_dict[merging_fn]['repair']['loss'].append(repair_loss)
                res_dict[merging_fn]['repair']['acc'].append(repair_acc)

                # rescale
                Merge.merged_model = deepcopy(merged_model_backup)
                repaired_merged_model = repair(train_loader, base_model_merge_s,
                                            Merge.merged_model, device,
                                            name=config['model']['name'], variant='rescale',
                                            reset_bn=False) # reset bn in merger
                Merge.merged_model = repaired_merged_model
                reset_bn_stats(Merge, train_loader)
                rescale_loss, rescale_acc = validate(Merge, test_loader, criterion, device)
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
