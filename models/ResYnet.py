import torch
from torch.nn import init
import torch.nn as nn
import math

__all__ = ['ResNet', 'ResUnet']


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_bn=False, affine_par=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_bn

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_bn
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_bn
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='up',
                 BN_enable=True, norm_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        #self.conv = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
        #                                                            kernel_size=3, stride=1, padding=0, bias=False))
        self.conv = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                    kernel_size=3, stride=1, padding=0, bias=False))

        if self.BN_enable:
            #self.norm1 = norm_layer(mid_channels)
            self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        #self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        elif self.upsample_mode == 'up':
            self.upsample = nn.Upsample(scale_factor=2)
        #if self.BN_enable:
        #    self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        #if self.BN_enable:
        #    x = self.norm2(x)
        #x = self.relu2(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, width=1.0, in_channel=3, unet=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        self.unet = unet
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = int(64 * width)
        self.base = int(64 * width)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer0 = nn.Sequential(
            #nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False), # 64
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=2, padding=0, bias=False),  # 32
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True))
        self.maxpool = nn.Sequential(nn.ReplicationPad2d(1), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # comment out fc layer for unsupervised learning, will have another head
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.final = MLPHead1(256, 512, 256)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mode=0):
        outputs = {}
        module_list = [self.layer0, self.maxpool, self.layer1, self.layer2, self.layer3, self.avgpool]
        for idx in range(len(module_list)):
            x = module_list[idx](x)
            outputs['d{}'.format(idx + 1)] = x
        return outputs



class ResYnet(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResYnet, self).__init__()

        filters = [64, 64, 128, 256, 512]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        norm_layer = nn.BatchNorm2d

        self.encoder1 = ResNet(BasicBlock, layer, width=0.5, in_channel=in_channel, unet=True, norm_layer=norm_layer)
        self.encoder2 = ResNet(BasicBlock, layer, width=0.5, in_channel=in_channel, unet=True, norm_layer=norm_layer)

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        outputs1 = self.encoder1(x1, mode=1)
        outputs2 = self.encoder2(x2, mode=0)
        e11 = outputs1['d1']
        e21 = outputs2['d1']
        e1 = torch.cat((e11, e21), dim=1)
        e12 = outputs1['d3']
        e22 = outputs2['d3']
        e2 = torch.cat((e12, e22), dim=1)
        e13 = outputs1['d4']
        e23 = outputs2['d4']
        e3 = torch.cat((e13, e23), dim=1)
        e14 = outputs1['d5']
        e24 = outputs2['d5']
        e4 = torch.cat((e14, e24), dim=1)
        center = self.center(e4)
        d2 = self.decoder1(torch.cat([center, e3], dim=1))
        d3 = self.decoder2(torch.cat([d2, e2], dim=1))
        d4 = self.decoder3(torch.cat([d3, e1], dim=1))
        return d4, outputs1['d6']


class ResYnet_(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResYnet_, self).__init__()

        filters = [32, 32, 64, 128, 256]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        norm_layer = nn.BatchNorm2d

        self.encoder1 = ResNet(BasicBlock, layer, width=0.5, in_channel=in_channel, unet=True, norm_layer=norm_layer)
        self.encoder2 = ResNet(BasicBlock, layer, width=0.5, in_channel=in_channel, unet=True, norm_layer=norm_layer)

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        outputs1 = self.encoder1(x1, mode=1)
        outputs2 = self.encoder2(x2, mode=0)
        e11 = outputs1['d1']
        e21 = outputs2['d1']
        # e1 = torch.cat((e11, e21), dim=1)
        e1 = e11 - e21
        e12 = outputs1['d3']
        e22 = outputs2['d3']
        # e2 = torch.cat((e12, e22), dim=1)
        e2 = e12 - e22
        e13 = outputs1['d4']
        e23 = outputs2['d4']
        # e3 = torch.cat((e13, e23), dim=1)
        e3 = e13- e23
        e14 = outputs1['d5']
        e24 = outputs2['d5']
        # e4 = torch.cat((e14, e24), dim=1)
        e4 = e14 - e24
        center = self.center(e4)
        d2 = self.decoder1(torch.cat([center, e3], dim=1))
        d3 = self.decoder2(torch.cat([d2, e2], dim=1))
        d4 = self.decoder3(torch.cat([d3, e1], dim=1))
        return d4, outputs1['d6']