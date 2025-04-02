from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
import os
import torch.utils.model_zoo as model_zoo


class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        f = nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        return x * f


class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        f = nn.functional.relu6(x + 3., inplace=self.inplace) / 6.
        return f


class SeModule(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SeModule, self).__init__()
        self.se_reduce = nn.Conv2d(in_channels, int(in_channels * se_ratio), kernel_size=1, stride=1, padding=0)
        self.se_expand = nn.Conv2d(int(in_channels * se_ratio), in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        s = nn.functional.adaptive_avg_pool2d(x, 1)
        s = self.se_expand(nn.functional.relu(self.se_reduce(s), inplace=True))
        return x * s.sigmoid()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = hswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, out_channel // reduction, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channel // reduction, out_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, kernel_size // 2)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = out * self.se(out)
        out += self.shortcut(x)
        out = nn.functional.relu(out, inplace=True)
        return out


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__()  #

        self.conv1 = ConvBlock(3, 16, 3, 2, 1)  # 1/2
        self.bottlenecks = nn.Sequential(
            ResidualBlock(16, 16, 3, 1, False),
            ResidualBlock(16, 24, 3, 2, False),  # 1/4
            ResidualBlock(24, 24, 3, 1, False),
            ResidualBlock(24, 40, 5, 2, True),  # 1/8
            ResidualBlock(40, 40, 5, 1, True),
            ResidualBlock(40, 40, 5, 1, True),
            ResidualBlock(40, 80, 3, 2, False),  # 1/16
            ResidualBlock(80, 80, 3, 1, False),
            ResidualBlock(80, 80, 3, 1, False),
            ResidualBlock(80, 112, 5, 1, True),
            ResidualBlock(112, 112, 5, 1, True),
            ResidualBlock(112, 160, 5, 2, True),  # 1/32
            ResidualBlock(160, 160, 5, 1, True),
            ResidualBlock(160, 160, 5, 1, True)
        )
        self.conv2 = ConvBlock(160, 960, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(960, 1280),
            nn.BatchNorm1d(1280),
            nn.Hardswish(inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bottlenecks(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv3(pretrained=True, downsample_factor=16):
    model = MobileNetV3Large (num_classes=1000)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model

