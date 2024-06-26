import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Sequential(nn.ReLU(), nn.ReplicationPad2d(padding), nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=0, dilation=dilation, bias=False))
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, BatchNorm):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]
        self.temperature = 0.8

        self.aspp1 = _ASPPModule(inplanes, 32, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 32, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 32, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 32, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 32, 1, stride=1, bias=False),
                                             BatchNorm(32),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(160, 64, 1, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        self.feature_select = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self._init_weight()

    def forward(self, x):
        cls_num = torch.from_numpy(np.array([[0], [1]], dtype=np.float32)).cuda()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        n, c, h, w = x.shape
        output = self.feature_select(x)
        clsmap = output
        output = F.gumbel_softmax(output, tau=self.temperature, dim=1, hard=False)
        output = rearrange(output, 'i j h w -> i j (h w)').contiguous()
        output = rearrange(output, 'i j hw -> i hw j').contiguous()
        output = torch.einsum("ijk,kl->ijl", output, cls_num).squeeze()
        output = output.contiguous().view(n, h, w)
        return output, x, clsmap


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(inplanes, BatchNorm):
    return ASPP(inplanes, BatchNorm)