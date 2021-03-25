'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FPN_n(nn.Module):
    ndf = 516

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=200)
        # self.fc1 = nn.Linear(in_channels, self.ndf)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, 40000)

    def forward(self, x):
        # Bottom-up
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = F.relu(self.bn3(out))
        out = self.conv4(out)
        out = F.relu(self.bn4(out))
        out = self.conv5(out)
        out = F.relu(self.bn5(out))
        # print(out)
        out = out.reshape(out.shape[0], 1, 32)
        # print(out)
        feature = self.fc(out)
        feature = feature.reshape(out.shape[0], 1, 200, 200)

        return feature


def Fpn_n():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN_n(1)


def test():
    net = Fpn_n()
    print(net)
    fms = net(torch.randn(1, 1, 200, 200))
    print(fms)
    for fm in fms:
        print(fm.size())


if __name__ == "__main__":
    test()
