from math import sqrt
import torch.nn.functional as F
import torch
from torch import nn


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    def __init__(self, in_chans=3, upscale=2, img_range=1.) -> None:
        super(VDSR, self).__init__()
        self.upscale = upscale
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )
        
        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, in_chans, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(x.shape[-2] * self.upscale, x.shape[-1] * self.upscale),
            mode="bicubic"
        )
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)
        out = out / self.img_range + self.mean
        return out
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))

if __name__ == '__main__':
    upscale = 4
    height = 64
    width = 48
    model = VDSR(
        in_chans=3,
        upscale=4
    )
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)