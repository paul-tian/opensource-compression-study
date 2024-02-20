r"""
AlexNet, the architecture comes from Pytorch official implementation
Both PyTorch implementation (StdAlexNet) and handmade implementation for pruning (AlexNet)
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress.pruning import PruningModule, MaskedLinear, MaskedConv2d


class AlexNet(PruningModule):
    r"""
    This is the Prune-able version of AlexNet.

    The input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        dropout (float): Probability of an element to be zeroed using nn.Dropout(). Default: 0.5
        mask (bool): True will use prune-able layer, False will use standard nn.layer. Default: False
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000, dropout: float = 0.5, mask=False):
        super().__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.conv1 = conv2d(in_channels, 64, kernel_size=(11, 11), stride=4, padding=2, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = conv2d(64, 192, kernel_size=(5, 5), padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = conv2d(192, 384, kernel_size=(3, 3), padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv2d(384, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = conv2d(256, 256, kernel_size=(3, 3), padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.dropout1 = nn.Dropout(p=dropout)
        # self.linear1 = linear(256 * 6 * 6, 4096)  # input size: [223, 254] (e.g. 223x223, 240x240, 254x254).
        self.linear1 = linear(256, 4096)  # input size: [63, 94] (e.g. 63x63, 80x80, 94x94).
        self.relu1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear3 = linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)

        x = self.linear1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        x = F.log_softmax(x, dim=1)
        return x


class StdAlexNet(nn.Module):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    The avgpool layer (nn.AdaptiveAvgPool2d()) has been removed for reproduction purpose,
    as this layer is non-deterministic. <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>
    Thus, the input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        dropout (float): Probability of an element to be zeroed using nn.Dropout(). Default: 0.5
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict({
                'conv1': nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2, bias=False),
                'relu1': nn.ReLU(inplace=True),
                'maxpool1': nn.MaxPool2d(kernel_size=3, stride=2),

                'conv2': nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
                'relu2': nn.ReLU(inplace=True),
                'maxpool2': nn.MaxPool2d(kernel_size=3, stride=2),

                'conv3': nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
                'relu3': nn.ReLU(inplace=True),

                'conv4': nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
                'relu4': nn.ReLU(inplace=True),

                'conv5': nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                'relu5': nn.ReLU(inplace=True),
                'maxpool5': nn.MaxPool2d(kernel_size=3, stride=2)
            })
        )

        self.classifier = nn.Sequential(
            OrderedDict({
                'dropout1': nn.Dropout(p=dropout),
                'linear1': nn.Linear(256 * 6 * 6, 4096),  # input size: [223, 254] (e.g. 223x223, 240x240, 254x254).
                'relu1': nn.ReLU(inplace=True),

                'dropout2': nn.Dropout(p=dropout),
                'linear2': nn.Linear(4096, 4096),
                'relu2': nn.ReLU(inplace=True),

                'linear3': nn.Linear(4096, num_classes)
            })
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
