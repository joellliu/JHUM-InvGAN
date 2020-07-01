import torch
import torch.nn as nn
import numpy as np
from torch.nn import InstanceNorm2d, Conv2d, Upsample


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear'):
        super(UpSampleBlock, self).__init__()
        self.model = nn.Sequential(
            Upsample(scale_factor=scale_factor, mode=mode),
            Conv2d(in_channels, out_channels, 3, 1, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.out_channels = out_channels
        self.model = nn.Sequential(
            InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(),
            conv,
            InstanceNorm2d(out_channels, affine=False)  # normalize to (0, 1) Gaussian
        )

    def forward(self, x):
        x = self.model(x)
        out = []
        for i in range(self.out_channels):
            out += [x[:, i, :, :].unsqueeze(1)]
        return out


class FirstResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstResBlock, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 1, padding=1),
            InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, padding=1),
            )
        self.bypass = nn.Sequential(
            Conv2d(in_channels, out_channels, 1, 1, padding=0),
            InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class EncodingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingResBlock, self).__init__()
        conv1 = Conv2d(in_channels, out_channels, 3, 2, padding=1)
        conv2 = Conv2d(out_channels, out_channels, 3, 1, padding=1)
        bypass_conv = Conv2d(in_channels, out_channels, 1, 1, padding=0)
        self.model = nn.Sequential(
            InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(),
            conv1,
            InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            conv2,
            )
        self.bypass = nn.Sequential(
            bypass_conv,
            nn.AvgPool2d(2),
            InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class DecodingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cat_channels):
        super(DecodingResBlock, self).__init__()
        # cat_channels: number of channels of the features from encoding path
        self.up = UpSampleBlock(in_channels, int(in_channels/2))
        n_main = int(in_channels/2) + cat_channels
        self.main = nn.Sequential(
            InstanceNorm2d(n_main, affine=True),
            nn.ReLU(),
            Conv2d(n_main, out_channels, 3, 1, padding=1),
            InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, padding=1)
        )
        self.bypass = nn.Sequential(
            Conv2d(n_main, out_channels, 1, 1, padding=0),
            InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x, e):
        x = self.up(x)
        x = torch.cat((x, e), 1)
        return self.main(x) + self.bypass(x)


class Dense(nn.Module):
    def __init__(self, s, z_dim, z_channel, net_channel):
        # z_dim: dimension of latent code
        # z_channel: # of channels of latent code
        # net_channel: # of channels of input feature maps
        # s: size of input feature maps
        super(Dense, self).__init__()
        self.s = s
        self.z_dim = z_dim
        self.z_channel = z_channel
        self.net_channel = net_channel
        self.dense = nn.Linear(s**2*net_channel, z_channel*z_dim)

    def forward(self, x):
        x = x.view(-1, self.net_channel*self.s**2)
        x = self.dense(x).view(-1, self.z_channel, self.z_dim)
        return x


class Encoder(nn.Module):
    def __init__(self, s0=4, z_dim=512, z_channel=14, net_dim=512, size=256):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.z_channel = z_channel
        self.net_dim = net_dim

        base_filter = 32
        max_filter = 512
        filters = [min(base_filter*2**i, max_filter) for i in range(7)]
        
        # encoding path
        Enc = [FirstResBlock(3, filters[0])]
        Enc += [EncodingResBlock(filters[i], filters[i+1]) for i in range(6)]
        self.Enc = nn.ModuleList(Enc)
        
        # decoding path
        self.Dec = nn.ModuleList([DecodingResBlock(filters[i + 1], filters[i], filters[i]) for i in range(6)])  # from largest resolution to smallest

        # noise prediction
        Conv = [ConvBlock(filters[i], 2) for i in range(6)]
        Conv += [ConvBlock(filters[6], 1)] # single noise for the smallest feature map
        self.Conv = nn.ModuleList(Conv)  # from largest resolution to smallest

        # latent code prediction
        self.Dense = Dense(s0, self.z_dim, self.z_channel, filters[6])

    def forward(self, x):
        e = []
        d = []
        # encoding path
        for encoder in self.Enc:
            x = encoder(x)
            e += [x]
        # latent code prediction
        latent = self.Dense(x)
        # noise prediction
        noises = self.Conv[6](x)
        # decoding path
        for i in reversed(range(6)): # from smallest resolution to largest
            x = self.Dec[i](x, e[i])
            d += [x]
            # noise prediction
            noises += self.Conv[i](x)
        return latent, noises






