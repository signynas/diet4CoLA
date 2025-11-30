import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(channels, groups=8):
    groups = min(groups, channels)
    groups = max(1, groups)
    return nn.GroupNorm(groups, channels)


class BaseConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        # GroupNorm because of small batch size
        self.norm1 = get_norm(in_ch)
        self.norm2 = get_norm(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + skip
    

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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = BaseConv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    
class UpConvResidual(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BaseConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # In case of odd spatial sizes, align skip to x
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.concat([x, skip], dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, base_ch=32, in_ch = 2, out_ch=1):
        super().__init__()

        self.name = 'UNet'
        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        self.layers = nn.Sequential(
            # Down
            BaseConv(base_ch, base_ch),
            DownConv(base_ch, base_ch * 2),
            DownConv(base_ch * 2, base_ch * 4),
            DownConv(base_ch * 4, base_ch * 8),

            # Bottleneck
            BaseConv(base_ch * 8, base_ch * 8),
            BaseConv(base_ch * 8, base_ch * 8),

            # Up
            UpConv(base_ch * 8, base_ch * 4),
            UpConv(base_ch * 4, base_ch * 2),
            UpConv(base_ch * 2, base_ch),
            BaseConv(base_ch, base_ch)
        )
            
    def forward(self, x):
        x = self.proj_in(x)
        x = self.layers(x)
        x = self.proj_out(x)

        return x