import torch
import torch.nn as nn
from models import common


class EDSR(nn.Module):
    def __init__(self, n_resblock=20, n_feats=64 ,res_scale=1, upscale=4, kernel_size=3, in_chans=3):
        super(EDSR, self).__init__()
        
        conv = conv=common.default_conv
        scale = upscale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv(in_chans, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_chans, kernel_size)
        ]

        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 


if __name__ == '__main__':
    upscale = 4
    height = 64
    width = 48
    model = EDSR(
        in_chans=3,
        upscale=4
    )
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)