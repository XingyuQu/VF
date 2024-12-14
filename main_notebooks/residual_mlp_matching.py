import sys
sys.path.append('..')

from source.utils.connect import eval_line
from source.utils.data_funcs import load_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from source.utils.logger import Logger
from source.utils.weight_matching import permutation_spec_from_axes_to_perm, PermutationSpec, get_wm_perm,  apply_permutation
import torch.nn.functional as F
from source.utils.utils import AverageMeter, ProgressMeter, \
    Summary, accuracy
import argparse
import time


class ResMLP(nn.Module):
    def __init__(self, in_channels=1, use_bias=True, num_classes=10, num_layers=4, w=16, use_residual=True):
        super().__init__()
        self.w = w
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_bias = use_bias
        # use a loop to create attributes called self.fc0, 1, 2, 3, ...
        for i in range(self.num_layers-1):
            setattr(self, f'fc{i}', nn.Linear(self.w if i > 0 else in_channels*28*28, self.w, bias=use_bias))
            if i > 0:
                res_matrix = torch.eye(self.w)
                if not use_residual:
                    res_matrix.fill_(0)
                self.register_buffer(f'res_{i}', res_matrix)
            # setattr(self, f'relu{i}', nn.ReLU())
        setattr(self, f'fc{self.num_layers-1}', nn.Linear(self.w, num_classes, bias=use_bias))
 
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = input.view(input.size(0), -1)
        for i in range(self.num_layers-1):
            input = out
            out = getattr(self, f'fc{i}')(out)
            if i > 0:
                res_matrix = getattr(self, f'res_{i}')
                out = out + F.linear(input, res_matrix)
            # out = getattr(self, f'relu{i}')(out)
        out = getattr(self, f'fc{self.num_layers-1}')(out)
        return out


def train(train_loader, model, criterion, optimizer,
          epoch, device, sampler, config, scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data = data.to(device)
        target = target.to(device)
        # compute output
        output_full = model(data)
        loss_full = criterion(output_full, target)
        counter = 1

        loss = loss_full
        loss /= counter

        # measure accuracy and record loss
        acc1 = accuracy(output_full, target, topk=(1,))
        losses.update(loss_full.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.print_freq == 0:
            progress.display(i + 1)
    return losses.avg, top1.avg, batch_time.avg


def validate(val_loader, model, criterion, device, config):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (data, target) in enumerate(loader):
                i = base_progress + i
                data = data.to(device)
                target = target.to(device)

                # compute output
                output = model(data)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), data.size(0))
                top1.update(acc1[0].item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    if config.model == 'simple_mlp':
        criterion = nn.MSELoss()

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return losses.avg, top1.avg, batch_time.avg


def mlp_permutation_spec(num_hidden_layers: int, bias=True, match_residual=True) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same
    weight array."""
    assert num_hidden_layers >= 1
    if bias:
        bias_hidden = {f"fc{i}.bias": (f"P_{i}", )
                       for i in range(num_hidden_layers)}
        bias_last = {f"fc{num_hidden_layers}.bias": (None, )}
    else:
        bias_hidden, bias_last = {}, {}

    if match_residual:
        return permutation_spec_from_axes_to_perm({
            "fc0.weight": ("P_0", None),
            **{f"fc{i}.weight": (f"P_{i}", f"P_{i-1}")
            for i in range(1, num_hidden_layers)},
            **{f"res_{i}": (f"P_{i}", f"P_{i-1}")
            for i in range(1, num_hidden_layers)},
            **bias_hidden,
            f"fc{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            **bias_last,
        })
    else:
        return permutation_spec_from_axes_to_perm({
            "fc0.weight": ("P_0", None),
            **{f"fc{i}.weight": (f"P_{i}", f"P_{i-1}")
            for i in range(1, num_hidden_layers)},
            **bias_hidden,
            f"fc{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            **bias_last,
        })

# python residual_mlp_matching.py --use_residual --repeated_runs 20 --save_path resmlp_wm_results.pth
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--repeated_runs', type=int, default=1)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    Logger.setup_logging()
    logger = Logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class config:
        model = 'mlp'
        dataset = 'mnist'

        print_freq = 100
        n = 11 # number of points on the line
        path = '../data' # path to dataset
        special_init = None

    # load data
    trainset, testset = load_data(config.path, 'mnist')
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    # model parameters
    use_bias = True
    num_classes = 10
    num_layers = 5
    w = 32
    use_residual = args.use_residual

    # run
    res_dict = {}
    for run in range(args.repeated_runs):
        model_1, model_2 = ResMLP(1, use_bias, num_classes, num_layers=num_layers, w=w, use_residual=use_residual).to(device), ResMLP(1, use_bias, num_classes, num_layers=num_layers, w=w, use_residual=use_residual).to(device)
        optimizer_1, optimizer_2 = torch.optim.Adam(model_1.parameters(), lr=0.01, weight_decay=1e-4), torch.optim.Adam(model_2.parameters(), lr=0.01, weight_decay=1e-4)  # use a small lr
        criterion = nn.CrossEntropyLoss()

        loss_s = [[], []]
        acc_s = [[], []]
        # training
        for epoch in range(20):
            loss_1, acc_1, _ = train(trainloader, model_1, criterion, optimizer_1, epoch, device, None, config)
            loss_2, acc_2, _ = train(trainloader, model_2, criterion, optimizer_2, epoch, device, None, config)
            loss_s[0].append(loss_1)
            loss_s[1].append(loss_2)
            acc_s[0].append(acc_1)
            acc_s[1].append(acc_2)
        lmc_stat = eval_line(model_1, model_2, testloader, criterion, device, config, False, n=11, name='mlp')

        # match
        match_residual = True
        ps = mlp_permutation_spec(num_layers-1, use_bias, match_residual)
        perm_2_wm = get_wm_perm(ps, model_1.state_dict(), model_2.state_dict(), device=device)
        sd_2_wm = apply_permutation(ps, perm_2_wm, model_2.state_dict(), device)
        model_2_wm = deepcopy(model_2)
        model_2_wm.load_state_dict(sd_2_wm)
        # test wm mid model
        lmc_wm_stat = eval_line(model_1, model_2_wm, testloader, criterion, device, config, False, n=11, name='mlp')
        cur_res_dict = {
            'loss_s': loss_s,
            'acc_s': acc_s,
            'lmc_stat': lmc_stat,
            'lmc_wm_stat': lmc_wm_stat,
        }
        res_dict[run] = cur_res_dict

    torch.save(res_dict, args.save_path)


if __name__ == '__main__':
    main()
