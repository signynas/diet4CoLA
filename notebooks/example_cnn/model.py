import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # GroupNorm usually works better for small batch sizes
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))

        return skip + x


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2)
        self.conv = BaseConv(in_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = BaseConv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ResNet(nn.Module):
    def __init__(self, base_ch=32, in_ch=2, out_ch=2):
        super().__init__()

        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        # Twice up, twice down
        self.layers = nn.Sequential(
            BaseConv(base_ch, base_ch),
            DownConv(base_ch, base_ch * 2),
            DownConv(base_ch * 2, base_ch * 4),

            BaseConv(base_ch * 4, base_ch * 4),
            BaseConv(base_ch * 4, base_ch * 4),

            UpConv(base_ch * 4, base_ch * 2),
            UpConv(base_ch * 2, base_ch),
            BaseConv(base_ch, base_ch),
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.layers(x)
        x = self.proj_out(x)
        return x
