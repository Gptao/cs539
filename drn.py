import torch
import torch.nn as nn
import common


def make_model(opt):
    return DRN(opt)


class DRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)  # 4 ->2,4 phase=2
        n_blocks = opt.n_blocks  # 30
        n_feats = opt.n_feats  # 20
        kernel_size = 3

        act = nn.ReLU(True)
        # 先进行scale倍插值上采样
        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # 减去均值
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
                             ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)
        # 上采样体
        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        # 按阶段拼接网络
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs 通道数：输入80,输出3
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)  # in 80 out 3  & in 40 out 3
            )
        self.tail = nn.ModuleList(tail)
        '''tail:
        ModuleList(
        (0): Conv2d(80, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Conv2d(80, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): Conv2d(40, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        '''
        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        # [1,3,128,128]
        # upsample x to target sr size
        x = self.upsample(x)
        # [1,3,512,512]
        # preprocess
        x = self.sub_mean(x)  # ？？？？？？？
        # [1,3,512,512]
        x = self.head(x)  # head 和 tail是颜色通道卷积
        # [1,20,512,512]

        # down phases,保存下采样中间结果用于上采样拼接
        copies = []
        for idx in range(self.phase):  # 2
            copies.append(x)
            x = self.down[idx](x)
            # [1, 40, 256, 256] & [1, 80, 128, 128]

        # up phases 原图起点
        sr = self.tail[0](x)
        # [1, 3, 128, 128]
        sr = self.add_mean(sr)
        # [1, 3, 128, 128]
        results = [sr]
        # 后面的结果
        for idx in range(self.phase):  # 2
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # [1, 40, 256, 256]             [1, 20, 512, 512]
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # [1, 80, 256, 256]             [1, 40, 512, 512
            # output sr imgs
            sr = self.tail[idx + 1](x)
            # [1, 80, 256, 256]             [1, 40, 512, 512]
            sr = self.add_mean(sr)
            # [1, 3, 256, 256]              [1, 3, 512, 512]

            results.append(sr)

        return results
