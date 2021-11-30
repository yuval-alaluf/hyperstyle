import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SharedWeightsHypernet(nn.Module):

    def __init__(self, f_size=3, z_dim=512, out_size=512, in_size=512, mode=None):
        super(SharedWeightsHypernet, self).__init__()
        self.mode = mode
        self.z_dim = z_dim
        self.f_size = f_size
        if self.mode == 'delta_per_channel':
            self.f_size = 1
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size * self.f_size * self.f_size)).cuda() / 40, 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.f_size * self.f_size)).cuda() / 40, 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)).cuda() / 40, 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)).cuda() / 40, 2))

    def forward(self, z):
        batch_size = z.shape[0]
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(batch_size, self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(batch_size, self.out_size, self.in_size, self.f_size, self.f_size)
        if self.mode == 'delta_per_channel':  # repeat per channel values to the 3x3 conv kernels
            kernel = kernel.repeat(1, 1, 1, 3, 3)
        return kernel
