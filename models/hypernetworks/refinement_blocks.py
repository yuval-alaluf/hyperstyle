import numpy as np
from torch import nn
from torch.nn import Conv2d, Sequential, Module

from models.encoders.helpers import SeparableBlock
from models.stylegan2.model import EqualLinear


# layer_idx: [kernel_size, in_channels, out_channels]
PARAMETERS = {
    0: [3, 512, 512],
    1: [1, 512, 3],
    2: [3, 512, 512],
    3: [3, 512, 512],
    4: [1, 512, 3],
    5: [3, 512, 512],
    6: [3, 512, 512],
    7: [1, 512, 3],
    8: [3, 512, 512],
    9: [3, 512, 512],
    10: [1, 512, 3],
    11: [3, 512, 512],
    12: [3, 512, 512],
    13: [1, 512, 3],
    14: [3, 512, 256],
    15: [3, 256, 256],
    16: [1, 256, 3],
    17: [3, 256, 128],
    18: [3, 128, 128],
    19: [1, 128, 3],
    20: [3, 128, 64],
    21: [3, 64, 64],
    22: [1, 64, 3],
    23: [3, 64, 32],
    24: [3, 32, 32],
    25: [1, 32, 3]
}
TO_RGB_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25]


class RefinementBlock(Module):

    def __init__(self, layer_idx, opts, n_channels=512, inner_c=256, spatial=16):
        super(RefinementBlock, self).__init__()
        self.layer_idx = layer_idx
        self.opts = opts
        self.kernel_size, self.in_channels, self.out_channels = PARAMETERS[self.layer_idx]
        self.spatial = spatial
        self.n_channels = n_channels
        self.inner_c = inner_c
        self.out_c = 512
        num_pools = int(np.log2(self.spatial)) - 1
        if self.kernel_size == 3:
            num_pools = num_pools - 1
        self.modules = []
        self.modules += [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*self.modules)

        if layer_idx in TO_RGB_LAYERS:
            self.output = Sequential(
                Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.output = Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1, stride=1,
                                            padding=0))

    def forward(self, x):
        x = self.convs(x)
        x = self.output(x)
        if self.layer_idx in TO_RGB_LAYERS:
            x = x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            x = x.view(-1, self.out_channels, self.in_channels)
            x = x.unsqueeze(3).repeat(1, 1, 1, self.kernel_size).unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size)
        return x


class HyperRefinementBlock(Module):
    def __init__(self, hypernet, n_channels=512, inner_c=128, spatial=16):
        super(HyperRefinementBlock, self).__init__()
        self.n_channels = n_channels
        self.inner_c = inner_c
        self.out_c = 512
        num_pools = int(np.log2(spatial))
        modules = [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(self.out_c, self.out_c, lr_mul=1)
        self.hypernet = hypernet

    def forward(self, features):
        code = self.convs(features)
        code = code.view(-1, self.out_c)
        code = self.linear(code)
        weight_delta = self.hypernet(code)
        return weight_delta


class RefinementBlockSeparable(Module):

    def __init__(self, layer_idx, opts, n_channels=512, inner_c=256, spatial=16):
        super(RefinementBlockSeparable, self).__init__()
        self.layer_idx = layer_idx
        self.kernel_size, self.in_channels, self.out_channels = PARAMETERS[self.layer_idx]
        self.spatial = spatial
        self.n_channels = n_channels
        self.inner_c = inner_c
        self.out_c = 512
        num_pools = int(np.log2(self.spatial)) - 1
        self.modules = []
        self.modules += [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*self.modules)

        self.opts = opts
        if self.layer_idx in TO_RGB_LAYERS:
            self.output = Sequential(Conv2d(self.out_c, self.in_channels * self.out_channels,
                                            kernel_size=1, stride=1, padding=0))
        else:
            self.output = Sequential(SeparableBlock(input_size=self.out_c,
                                                    kernel_channels_in=self.in_channels,
                                                    kernel_channels_out=self.out_channels,
                                                    kernel_size=self.kernel_size))

    def forward(self, x):
        x = self.convs(x)
        x = self.output(x)
        if self.layer_idx in TO_RGB_LAYERS:
            x = x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return x