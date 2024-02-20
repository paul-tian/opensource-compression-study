r"""
ResNet model architecture from the
`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

The architecture comes from Pytorch official implementation
Both PyTorch implementation (StdResNet18) and handmade implementation for pruning (ResNet18)
"""

from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from compress.pruning import PruningModule, MaskedLinear, MaskedConv2d


class Residual(PruningModule):
    """The Residual block of ResNet.
    """

    def __init__(self, input_channels, num_channels, use_1x1conv: bool = False, strides: int = 1, mask: bool = False):
        super().__init__()
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.conv1 = conv2d(input_channels, num_channels, kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = conv2d(input_channels, num_channels, kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet18(PruningModule):
    r"""
    This is the Prune-able version of ResNet18.

    The input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        mask (bool): True will use prune-able layer, False will use standard nn.layer(). Default: False
    """

    def __init__(self, in_size: int = 1 * 128, exp_size: int = 33,
                 in_channels: int = 9, num_classes: int = 6,
                 init_weights: bool = True, mask: bool = False):

        super().__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.exp_size = exp_size
        self.fc = nn.Linear(in_size, exp_size*exp_size, bias=False)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.ln = nn.LayerNorm([in_channels, 1, exp_size*exp_size])

        self.b1 = nn.Sequential(
            conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        def resnet_block(input_channels, num_channels, num_residuals,
                         first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2, mask=mask))
                else:
                    blk.append(Residual(num_channels, num_channels, mask=mask))
            return blk

        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.linear1 = linear(2048, num_classes)  # input size: [33, 64] (e.g. 33x33, 40x40, 64x64).

        if init_weights:
            for m in self.modules():
                if isinstance(m, conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.fc(x)
        # x = self.bn(x)
        x = self.ln(x)
        x = x.view(x.size()[0], x.size()[1], self.exp_size, self.exp_size)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=1)
        return x

