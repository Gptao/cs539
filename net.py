from torch import nn


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
    def __init__(self):
        super(DownBlock, self).__init__()
        self.dual_module = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(20, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.dual_module(x)
        x = x.squeeze(0).permute(1, 2, 0)
        return x


class Downnet(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(Downnet, self).__init__()

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
