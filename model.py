import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# Hyper U-Net(+CBAM) model
class UNetCBAM(nn.Module):
    def __init__(self, use_cbam=True):
        super().__init__()

        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.cbam1 = CBAM(64)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.cbam2 = CBAM(128)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.cbam3 = CBAM(256)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.cbam4 = CBAM(512)

        if not use_cbam:
          self.cbam1 = nn.Identity()
          self.cbam2 = nn.Identity()
          self.cbam3 = nn.Identity()
          self.cbam4 = nn.Identity()

        self.bottleneck = ConvBlock(512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.hyper_up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # from encoder 2
        self.hyper_up_dec2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # from decoder 2

        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 128 + 128, 128, kernel_size=3, padding=1),  # d1 + up(conv2) + up(d2)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.cbam1(x1)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        x2 = self.cbam2(x2)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        x3 = self.cbam3(x3)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        x4 = self.cbam4(x4)
        p4 = self.pool4(x4)

        bottleneck = self.bottleneck(p4)

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        h2 = self.hyper_up2(x2)
        h_dec2 = self.hyper_up_dec2(d2)

        out = self.final_conv(torch.cat([d1, h2, h_dec2], dim=1))
        return out
    