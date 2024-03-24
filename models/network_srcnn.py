import torch
from torch import nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, in_chans=3, upscale=2, img_range=1.):
        super(SRCNN, self).__init__()
        self.upscale = upscale
        self.img_range = img_range
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, in_chans, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(x.shape[-2] * self.upscale, x.shape[-1] * self.upscale),
            mode="bicubic"
        )
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x / self.img_range + self.mean
        return x

if __name__ == '__main__':
    upscale = 4
    height = 64
    width = 48
    model = SRCNN(
        in_chans=3,
        upscale=4
    )
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)