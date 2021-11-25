"""
Reference

https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA
"""
import torch.nn as nn
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class FeatureExtractor(Backbone):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.ca1 = ChannelAttention(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca2 = ChannelAttention(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.ca3 = ChannelAttention(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

        self._out_features = 2048

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.bn1(self.conv1(x))

        out_ca1 = []
        x = F.relu(x)
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = self.bn2(self.conv2(x))
        out_ca2 = self.ca2(x)
        x = F.relu(x * out_ca2)
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        out_ca3 = self.ca3(x)
        x = F.relu(x * out_ca3)
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x, [out_ca1, out_ca2, out_ca3]


@BACKBONE_REGISTRY.register()
def cnn_digit5_m3sda_ca(**kwargs):
    """
    This architecture was used for the Digit-5 dataset in:

        - Peng et al. Moment Matching for Multi-Source
        Domain Adaptation. ICCV 2019.
    """
    return FeatureExtractor()
