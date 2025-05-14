import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.down1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.down4(pool3)
        pool4 = self.pool4(conv4)

        bottleneck = self.bottleneck(pool4)

        up4 = self.up4(bottleneck)
        up4 = torch.cat([up4, conv4], dim=1)
        conv_up4 = self.up_conv4(up4)

        up3 = self.up3(conv_up4)
        up3 = torch.cat([up3, conv3], dim=1)
        conv_up3 = self.up_conv3(up3)

        up2 = self.up2(conv_up3)
        up2 = torch.cat([up2, conv2], dim=1)
        conv_up2 = self.up_conv2(up2)

        up1 = self.up1(conv_up2)
        up1 = torch.cat([up1, conv1], dim=1)
        conv_up1 = self.up_conv1(up1)

        final = self.final_conv(conv_up1)
        return final