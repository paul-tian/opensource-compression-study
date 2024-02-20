import math
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

'''
This n tuple defenition is cpoied from PyTorch code:
https://github.com/pytorch/pytorch/blob/ddda7cfea521fa442ac117f0bc1bfdbf01ab0c63/torch/nn/modules/utils.py
'''
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class PruningModule(Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        r"""
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        alive_parameters = []
        for name, p in self.named_parameters():
            # We do not prune bias term
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}')

        # Prune the weights and mask
        # Note that module here is the layer
        for name, module in self.named_modules():
            try:
                module.prune(threshold=percentile_value)
            except:
                continue

    def prune_by_threshold(self, s=0.25):
        r"""
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            try:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)
            except:
                continue


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class MaskedConv2d(Module):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
            .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                            \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                            \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                             (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_channels, in_channels, kernel_size[0], kernel_size[1])
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.mask = Parameter(torch.ones(out_channels, in_channels, kernel_size[0], kernel_size[1]),
                              requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        r'''
        This part is copied from old version of Pytorch
        https://github.com/pytorch/pytorch/blob/e8536c08a16b533fe0a9d645dd4255513f9f4fdd/torch/nn/modules/conv.py
        Which is committed on July 25, 2018
        After then, the reset_parameters(self) has been modified to use kaiming_norm
        For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        '''
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, bias=self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
