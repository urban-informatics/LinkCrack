import torch.nn.functional as F
from torch import nn
import torch


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=(1, 1), shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation[0], padding=dilation[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation[1], padding=dilation[1], bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, scale, upsample_mode='bilinear', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        self.scale = scale

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                              bias=True)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, down_inp, up_inp):
        x = torch.cat([down_inp, up_inp], 1)
        x = self.conv1(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=True)
        x = self.conv2(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu(x)
        return x

class LinkCrack(nn.Module):

    def __init__(self):
        super(LinkCrack, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._res1_shorcut = nn.Sequential(
            nn.Conv2d(64, 64, 1, 2, bias=False),
            nn.BatchNorm2d(64)
        )

        self.res1 = nn.Sequential(
            ResidualBlock(64, 64, stride=2, shortcut=self._res1_shorcut),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
        )


        self._res2_shorcut = nn.Sequential(
            nn.Conv2d(64, 64, 1, 2, bias=False),
            nn.BatchNorm2d(64)
        )

        self.res2 = nn.Sequential(
            ResidualBlock(64, 64, stride=2, shortcut=self._res2_shorcut),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
        )

        self._res3_shorcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.res3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, shortcut=self._res3_shorcut),
            ResidualBlock(128, 128),  # 1
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, dilation=(2,2)), # 【N 128 64 64】 2-dilated
            ResidualBlock(128, 128,dilation=(2,2)),
            ResidualBlock(128, 128, dilation=(2,2)),
        )

        self._res4_shorcut = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.res4 = nn.Sequential(
            ResidualBlock(128, 128,dilation=(2,2), shortcut=self._res4_shorcut),
            ResidualBlock(128, 128, dilation=(4,4)),  # 4-dialted
            ResidualBlock(128, 128, dilation=(4,4)),
        )
        self.dec4 = DecoderBlock(in_channels=128+128, mid_channels=128, out_channels=64, scale=2)
        self.dec3 = DecoderBlock(in_channels=64+64, mid_channels=64, out_channels=64, scale=2)
        self.dec2 = DecoderBlock(in_channels=64+64, mid_channels=64, out_channels=64,scale=2)

        self.mask = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
        )

        self.link = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
        )

    def forward(self, x):

        x = self.pre(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.dec4(x4,x3)
        x6 = self.dec3(x5, x2)
        x7 = self.dec2(x6,x1)
        mask = self.mask(x7)
        link = self.link(x7)
        return mask, link
