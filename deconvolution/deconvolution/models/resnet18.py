from torch import nn
from deconv import FastDeconv, DeLinear
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), deconv=False):
        super(ResidualBlock, self).__init__()
        self.deconv = deconv
        self.shortcut = nn.Sequential()
        if deconv:
            self.conv = nn.Sequential(
                FastDeconv(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
                nn.ReLU(inplace=True),
                FastDeconv(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            )
            if stride[0] != 1:
                self.shortcut = nn.Sequential(
                    FastDeconv(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if stride[0] != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet18 simple version for 32x32 input
class ResNet18(nn.Module):
    def __init__(self, n_classes=10, deconv=False, delinear=False):
        super(ResNet18, self).__init__()
        self.deconv = deconv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_conv = self.res_layers()
        if delinear:
            self.fc = DeLinear(512, n_classes, block=512)
        else:
            self.fc = nn.Linear(512, n_classes)

    def res_layers(self):
        layers = []
        layers += [
            ResidualBlock(64, 64, deconv=self.deconv),  # 64, 32, 32
            ResidualBlock(64, 64, deconv=self.deconv)  # 64, 32, 32
        ]
        layers += [
            ResidualBlock(64, 128, (2, 2), deconv=self.deconv),  # 128, 16, 16
            ResidualBlock(128, 128, deconv=self.deconv)  # 128, 16, 16
        ]
        layers += [
            ResidualBlock(128, 256, (2, 2), deconv=self.deconv),  # 256, 8, 8
            ResidualBlock(256, 256, deconv=self.deconv)  # 256, 8, 8
        ]
        layers += [
            ResidualBlock(256, 512, (2, 2), deconv=self.deconv),  # 512, 4, 4
            ResidualBlock(512, 512, deconv=self.deconv)  # 512, 4, 4
        ]
        layers += [nn.AdaptiveAvgPool2d(1)]  # 512, 1, 1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_conv(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
