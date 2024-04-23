import torch
from .vit import vit_tiny
from .biconvlstm import ConvLSTM
from torch import nn
from torch.autograd import Variable


class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DinoFeaturizer(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False, dim=70, img_size=[128], patch_size=8, pretrained_weights='', dropout_feat=False):
        super(DinoFeaturizer, self).__init__()

        self.hidden_channels = hidden_channels
        self.ouput_channels = ouput_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.temperature = 0.8

        convlstm_layer = []
        for i in range(num_layers):
            layer = ConvLSTM(img_size_list[i],
                             input_channels,
                             hidden_channels[i],
                             kernel_size[i],
                             0.2, 0.,
                             batch_first=True,
                             bias=True,
                             peephole=False,
                             layer_norm=False,
                             return_sequence=True,
                             bidirectional=self.bidirectional)
            convlstm_layer.append(layer)
            input_channels = hidden_channels[i] * (2 if self.bidirectional else 1)

        self.convlstm_layer = torch.nn.ModuleList(convlstm_layer)
        self.model = vit_tiny(img_size=img_size, patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)
        self.dropout_feat = dropout_feat

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            state_dict = state_dict["student"]
            msg = self.model.load_state_dict({k.replace('backbone.', ''): v for k, v in state_dict.items()},
                                             strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

        self.active = nn.Sequential(nn.Conv2d(192, dim, (1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(dim, int(0.5 * dim), (1, 1)),
                                    nn.ReLU())

        self.contrast = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, dim, kernel_size=1, stride=1,
                                                padding=0, bias=False))


    def forward(self, input_tensor):
        input_tensor = self.model.get_patch_feature(input_tensor)
        feat = self.dropout(input_tensor)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        code = self.contrast(input_tensor)

        return feat, code


if __name__ == '__main__':
    # gradient check
    layer_num = 1
    convlstm = DinoFeaturizer([[64, 64], ], input_channels=128, hidden_channels=[128, 128], ouput_channels=2,
                               kernel_size=[[3, 3], ], num_layers=layer_num,
                               bidirectional=True).cuda()
    loss_fn = torch.nn.MSELoss()

    # batchsize, timesteps, features, width, height
    input = Variable(torch.randn(1, 2, 128, 64, 64)).cuda()
    target = Variable(torch.randn(1, 2, 64, 64)).double().cuda()

    output = convlstm(input)
    output = output.double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)