from torch import nn
import numpy as np


class ZSSRNet(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(ZSSRNet, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.relu = nn.ReLU()

    def forward(self, y):
        # [1, 3, h, w]
        # 中间结果为[1,64,h,w]
        # print('fuck', y.shape)
        x = y.permute(2, 0, 1).unsqueeze(0)  # 图像 HWC 转 1CHW
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        x = x.squeeze(0).permute(1, 2, 0)  # 1CHW 转 图像 HWC

        return x + y


class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval  # 0.2

        if nFeat is None:
            nFeat = opt.n_feats  # 20

        if in_channels is None:
            in_channels = opt.n_colors  # 3

        if out_channels is None:
            out_channels = opt.n_colors  # 3

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.dual_module(x)
        return x
