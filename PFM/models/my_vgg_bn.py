# import torch
from torch import nn
# import torchvision.transforms as T

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG_BN(nn.Module):
    def __init__(self, vgg_name, w=1, num_classes=10):
        super(VGG_BN, self).__init__()
        self.w = w
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Sequential(nn.Linear(self.w*512, num_classes))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(self.w*x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return nn.Sequential(*layers)
    
def my_vgg11_bn(w=1, num_classes=10):
    return VGG_BN('VGG11', w, num_classes=num_classes)

def my_vgg16_bn(w=1, num_classes=10):
    return VGG_BN('VGG16', w, num_classes=num_classes)
