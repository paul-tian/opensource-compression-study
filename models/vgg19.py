r"""
VGG model architecture from the
`"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_ paper.

The architecture comes from Pytorch official implementation
Both PyTorch implementation (StdVGG19) and handmade implementation for pruning (VGG19)
"""

from typing import Union, List, Dict, Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress.pruning import PruningModule, MaskedLinear, MaskedConv2d


class VGG19(PruningModule):
    r"""
    This is the Prune-able version of VGG16.

    The input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        dropout (float): Probability of an element to be zeroed using nn.Dropout(). Default: 0.5
        mask (bool): True will use prune-able layer, False will use standard nn.layer(). Default: False
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5,
                 mask=False):
        super().__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.conv1_1 = conv2d(in_channels, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.linear1 = linear(512 * 7 * 7, 4096)  # input size: [224, 255] (e.g. 224x224, 240x240, 255x255).
        self.linear1 = linear(512, 4096)  # input size: [32, 63] (e.g. 32x32, 40x40, 63x63).
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)

        self.linear2 = linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear3 = linear(4096, num_classes)

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
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)

        x = F.log_softmax(x, dim=1)
        return x


class VGG19BN(PruningModule):
    r"""
    This is the Prune-able version of VGG16 with Batch Normalization.

    The input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        dropout (float): Probability of an element to be zeroed using nn.Dropout(). Default: 0.5
        mask (bool): True will use prune-able layer, False will use standard nn.layer(). Default: False
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5,
                 mask=False):
        super().__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.conv1_1 = conv2d(in_channels, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.linear1 = linear(512 * 7 * 7, 4096)  # input size: [224, 255] (e.g. 224x224, 240x240, 255x255).
        self.linear1 = linear(512, 4096)  # input size: [32, 63] (e.g. 32x32, 40x40, 63x63).
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)

        self.linear2 = linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear3 = linear(4096, num_classes)

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
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.bn3_4(x)
        x = self.relu3_4(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.bn4_4(x)
        x = self.relu4_4(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu5_3(x)
        x = self.conv5_4(x)
        x = self.bn5_4(x)
        x = self.relu5_4(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)

        x = F.log_softmax(x, dim=1)
        return x


class StdVGG19(nn.Module):
    r"""
    This is the PyTorch official implementation of VGG.

    The avgpool layer (nn.AdaptiveAvgPool2d()) has been removed for reproduction purpose, as this layer is
    non-deterministic. <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>
    Thus, the input channel of the first layer of classifier is corresponding with input image size,
    and need to be manually chosen.

    Args:
        in_channels (int): Number of channels in the input image. Default: 1
        num_classes (int): Number of classification. Default: 10
        dropout (float): Probability of an element to be zeroed using nn.Dropout(). Default: 0.5
    """

    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),  # input size: [224, 255] (e.g. 224x224, 240x240, 255x255).
            nn.Linear(512, 4096),  # input size: [32, 63] (e.g. 32x32, 40x40, 63x63).
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


def make_layers(cfg: List[Union[str, int]], in_channels: int = 3, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = in_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> StdVGG19:
    model = StdVGG19(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=kwargs.pop('in_channels', 3)), **kwargs)
    return model


def stdvgg19(**kwargs: Any) -> StdVGG19:
    r"""VGG 19-layer model (configuration "E")
    """
    return _vgg("E", False, **kwargs)


def stdvgg19_bn(**kwargs: Any) -> StdVGG19:
    r"""VGG 19-layer model (configuration "E") with batch normalization
    """
    return _vgg("E", True, **kwargs)
