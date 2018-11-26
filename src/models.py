import torch
from torch import nn
from torch.nn import functional as F
import torchvision
# from zoo.resnet import *

archs = [
    'tmp',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'zfnet',
    'dpn68',
    'dpn92'
]


def fix_batchnorm_(m):
    # default momentum 0.1 causes large difference between train and eval modes.
    # increasing momentum helps
    if type(m) == torch.nn.BatchNorm2d:
        m.momentum = 0.5
    for m_ in m.children():
        fix_batchnorm_(m_)


def get_model(model_path, model_type, num_channels=4, num_classes=28):
    if model_type == 'resnet18':
        model = CustomResnet2(encoder_depth=18, num_classes=num_classes)
    elif model_type == 'resnet34':
        model = CustomResnet2(encoder_depth=34, num_classes=num_classes)
    elif model_type == 'tmp':
        model = CustomResnet(encoder_depth=18, num_classes=num_classes)
    elif model_type == 'resnet50':
        model = CustomResnet2(encoder_depth=50, num_classes=num_classes)
    elif model_type == 'resnet101':
        model = CustomResnet2(encoder_depth=101, num_classes=num_classes)
    elif model_type == 'zfnet':
        model = ZFNet(channels=num_channels, class_count=num_classes)
    elif model_type == 'dpn68':
        model = Dpn('dpn68')
    elif model_type == 'dpn92':
        model = Dpn('dpn92')
    else:
        raise NotImplementedError

    if model_path is not None:
        state = torch.load(str(model_path))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state, strict=False)
        
    fix_batchnorm_(model)

    if torch.cuda.is_available():
        return model.cuda()


class ZFNet(nn.Module):
    def __init__(self, channels, class_count):
        super(ZFNet, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

    def get_conv_net(self):
        layers = []

        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))


        return nn.Sequential(*layers)

    def get_fc_net(self):
        layers = []

        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(9216, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, self.class_count)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 9216)
        y = self.fc_net(y)
        return y



RESNET_ENCODERS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}


import pretrainedmodels


class Dpn(nn.Module):
    def __init__(self, model_name, num_classes=28):
        super().__init__()
        self.dpn = pretrainedmodels.__dict__[model_name]()

        input_block = self.dpn.features[0]
        original_conv = input_block.conv

        num_init_features = 64
        self.conv1 = nn.Conv2d(4, num_init_features, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        w = original_conv.weight
        w_size = w.size()

        self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(w_size[0], 1, w_size[2], w_size[3])), dim=1))

        input_block.conv = self.conv1

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        # dpn68, 832
        self.dpn.last_linear = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.dpn.forward(x)
        return x


class CustomResnet2(nn.Module):
    def __init__(self, encoder_depth=34, pretrained=True, num_classes=28):
        super().__init__()

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)
        self.in_planes = 64

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * (1 if encoder_depth in [18, 34] else 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    dpn = Dpn('dpn68')
    for m in dpn.modules():
        print(m)
    import numpy as np
    x = torch.from_numpy(np.zeros((1, 4, 224, 224), dtype=np.float32))
    dpn.forward(x)



