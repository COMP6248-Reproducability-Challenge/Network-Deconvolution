from torch import nn
from deconv import FastDeconv, DeLinear

class Vgg16(nn.Module):
    def __init__(self, n_classes=10, deconv=False, delinear=False):
        super(Vgg16, self).__init__()
        self.deconv = deconv
        self.conv = self.conv_layers()
        if delinear:
            self.fc = DeLinear(512, n_classes, block=512)
        else:
            self.fc = nn.Sequential(nn.Linear(512, n_classes))

    def conv_block(self, in_channels, out_channels, freeze=False, n_iter=5):
        if self.deconv:
            return nn.Sequential(FastDeconv(in_channels, out_channels, (3, 3), padding=(1, 1),
                                            freeze=freeze,
                                            n_iter=n_iter),
                                 nn.ReLU(inplace=True))
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def conv_layers(self):
        layers = []
        layers += [self.conv_block(3, 64, freeze=True, n_iter=15),
                   self.conv_block(64, 64),
                   nn.MaxPool2d((2, 2), (2, 2))]

        layers += [self.conv_block(64, 128),
                   self.conv_block(128, 128),
                   nn.MaxPool2d((2, 2), (2, 2))]

        layers += [self.conv_block(128, 256),
                   self.conv_block(256, 256),
                   self.conv_block(256, 256),
                   nn.MaxPool2d((2, 2), (2, 2))]

        layers += [self.conv_block(256, 512),
                   self.conv_block(512, 512),
                   self.conv_block(512, 512),
                   nn.MaxPool2d((2, 2), (2, 2))]

        layers += [self.conv_block(512, 512),
                   self.conv_block(512, 512),
                   self.conv_block(512, 512),
                   nn.MaxPool2d((2, 2), (2, 2))]

        layers += [nn.AdaptiveAvgPool2d(1)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x