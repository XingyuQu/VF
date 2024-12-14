"""Based on:
(open_lth)
"""
import torch.nn as nn
import torch.nn.functional as F


# TODO: different initialization methods?
class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, norm_layer, downsample=False):
            super(ResNet.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.norm1 = norm_layer(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.norm2 = norm_layer(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        f_in, f_out, kernel_size=1,
                        stride=2, bias=False),
                    norm_layer(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, plan, norm='bn', num_classes=None):
        super(ResNet, self).__init__()
        num_classes = num_classes or 10

        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f'Invalid norm: {norm}')

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv1 = nn.Conv2d(3, current_filters, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.norm1 = norm_layer(current_filters)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The subsequent blocks of the ResNet.
        segments = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            blocks = []
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters,
                                           norm_layer, downsample))
                current_filters = filters
            blocks = nn.Sequential(*blocks)
            segments.append(blocks)
        self.segments = nn.Sequential(*segments)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], num_classes)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.segments(out)
        out = self.avg_pool(out) # F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def get_model_from_name(model_name, num_classes=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has 16W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 1 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_1'.
        """

        norm = 'bn'

        name = model_name.split('_')
        W = 16 * int(name[-1][:-1]) if name[-1].endswith('x') else 16
        D = int(name[1][6:])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return ResNet(plan, norm, num_classes)
