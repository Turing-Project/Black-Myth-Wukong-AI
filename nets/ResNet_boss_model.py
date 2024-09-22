'''
这个是直接用的resnet50走预训练更好提取特征
'''
import torchdiffeq
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetblock, self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 4)
        )
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = x
        out = self.blockconv(x)
        if hasattr(self, 'shortcut'):
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet50_boss(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_boss, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.in_channels = 64
        # ResNet50中的四大层，每大层都是由ConvBlock与IdentityBlock堆叠而成
        self.layer1 = self.make_layer(ResNetblock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResNetblock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResNetblock, 64, 6, stride=2)
        self.layer4 = self.make_layer(ResNetblock, 256, 3, stride=2)

        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc_1 = nn.Linear(1024 * 5 * 5, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

    def make_layer(self, block, channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([32, 64, 100, 80])
        out = self.layer1(out)  # torch.Size([32, 256, 100, 80])
        out = self.layer2(out)  # torch.Size([32, 512, 50, 40])
        out = self.layer3(out)  # torch.Size([32, 256, 25, 20])
        out = self.layer4(out)  # torch.Size([32, 1024, 13, 10])
        out = self.avgpool(out)  # torch.Size([32, 1024, 6, 5])
        out = out.view(out.size(0), -1)  # torch.Size([32, 12288])
        out = self.fc_1(out)  # torch.Size([32, 512])
        embed = self.fc_2(out)  # torch.Size([32, 128])
        out = self.fc_3(embed)  # torch.Size([32, 5])

        return F.log_softmax(out, dim=1), embed