import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Blue(nn.Module):
    def __init__(self, inplane, plane):
        super(Blue, self).__init__()
        self.conv=nn.Conv2d(inplane, plane, kernel_size=3, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(plane)
    def forward(self, x):
        x=F.relu(self.bn(self.conv(x)))
        return x

class SegNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SegNet, self).__init__()
        self.inplane = 3
        self.encoder1 = self._make_layer([64, 64, 'M'])
        self.encoder2 = self._make_layer([128, 128, 'M'])
        self.encoder3 = self._make_layer([256, 256, 256, 'M'])
        self.encoder4 = self._make_layer([512, 512, 512, 'M'])
        self.encoder5 = self._make_layer([512, 512, 512, 'M'])

        self.decoder1 = self._make_layer([512, 512, 512])
        self.decoder2 = self._make_layer([256, 256, 256])
        self.decoder3 = self._make_layer([256, 256, 128])
        self.decoder4 = self._make_layer([128, 64])
        self.decoder5 = self._make_layer([64, num_classes])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        layers = []
        for m in planes:
            if isinstance(m, int):
                layers += [Blue(self.inplane, m)]
                self.inplane = m
            elif m == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, return_indices=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x, i1 = self.encoder1(x)
        x, i2 = self.encoder2(x)
        x, i3 = self.encoder3(x)
        x, i4 = self.encoder4(x)
        x, i5 = self.encoder5(x)

        x = self.decoder1(F.max_unpool2d(x, i5, kernel_size=2))
        x = self.decoder2(F.max_unpool2d(x, i4, kernel_size=2))
        x = self.decoder3(F.max_unpool2d(x, i3, kernel_size=2))
        x = self.decoder4(F.max_unpool2d(x, i2, kernel_size=2))
        x = self.decoder5(F.max_unpool2d(x, i1, kernel_size=2))

        x = F.sigmoid(x)
        return x


