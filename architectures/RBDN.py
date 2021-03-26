import torch
import torch.nn as nn
import torch.nn.functional as F


class RBDN(nn.Module):
    num_output = 64

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convB1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnB1_1 = nn.BatchNorm2d(64)
        self.poolB1 = nn.MaxPool2d(2, 2)

        self.convB2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnB2_1 = nn.BatchNorm2d(64)
        self.poolB2 = nn.MaxPool2d(2, 2)

        self.convB3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnB3_1 = nn.BatchNorm2d(64)
        self.poolB3 = nn.MaxPool2d(2, 2)

        self.convB3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnB3_2 = nn.BatchNorm2d(64)
        self.unpoolB3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconvB3_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.debnB3_1=nn.BatchNorm2d(64)
        self.convB2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnB2_2 = nn.BatchNorm2d(64)
        self.convB1_2 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bnB1_2 = nn.BatchNorm2d(self.num_output)
        self.conv2_1 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_1 = nn.BatchNorm2d(self.num_output)
        self.conv2_2 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_2 = nn.BatchNorm2d(self.num_output)
        self.conv2_3 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_3 = nn.BatchNorm2d(self.num_output)
        self.conv2_4 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_4 = nn.BatchNorm2d(self.num_output)
        self.conv2_5 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_5 = nn.BatchNorm2d(self.num_output)
        self.conv2_6 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_6 = nn.BatchNorm2d(self.num_output)
        self.conv2_7 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_7 = nn.BatchNorm2d(self.num_output)
        self.conv2_8 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_8 = nn.BatchNorm2d(self.num_output)
        self.conv2_9 = nn.Conv2d(in_channels, self.num_output, kernel_size=3, padding=1, stride=1)
        self.bn2_9 = nn.BatchNorm2d(self.num_output)
