import torch
import copy
import numpy as np
from .biconvlstm import ConvLSTM
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange, repeat


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


class ConvLSTMNetwork(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetwork, self).__init__()

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
        self.contrast = nn.Sequential(nn.ReLU(), nn.Conv2d(int(hidden_channels[-1] * 2), ouput_channels,
                                                           kernel_size=1, stride=1, padding=0, bias=False))
        self.extract = nn.Sequential(nn.Linear(1280, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))


    def forward(self, feats, token):
        ni, ci, hi, wi = feats.shape
        if len(token.shape) == 3:
            #print(token.shape)
            token = self.extract(token)
            #print(token.shape)
            token = repeat(token.unsqueeze(3), 'n c h w -> n c (repeat1 h) (repeat2 w)', repeat1=hi, repeat2=wi)
            out_pos = [torch.cat((feats[i, :, :, :],  token[i, :, :, :]), dim=0).unsqueeze(0) for i in range(ni)]
        elif len(token.shape) == 4:
            out_pos = [torch.cat((feats[i, :, :, :], token[i, :, :, :]), dim=0).unsqueeze(0) for i in range(ni)]
        else:
            print('is not support')
        input_tensor = torch.cat(out_pos, dim=0).unsqueeze(0)

        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        supfet = self.contrast(input_tensor)
        contst = nn.Sigmoid()(supfet)
        #output = F.gumbel_softmax(contst, tau=self.temperature, dim=1, hard=False)
        #output = rearrange(output, 'i j h w -> i j (h w)').contiguous()
        #output = rearrange(output, 'i j hw -> i hw j').contiguous()
        #output = torch.einsum("ijk,kl->ijl", output, cls_num).squeeze()
        #output = output.contiguous().view(b, n, h, w)

        return None, input_tensor, contst, supfet


if __name__ == '__main__':
    '''1
    convlstm_layer = []
        img_size_list=[(10, 10)]
        num_layers = 1              # number of layer
        input_channel = 96          # the number of electrodes in Utah array
        hidden_channels = [256]     # the output channels for each layer
        kernel_size = [(7, 7)]      # the kernel size of cnn for each layer
        stride = [(1, 1)]           # the stride size of cnn for each layer
        padding = [(0, 0)]          # padding size of cnn for each layer
        for i in range(num_layers):
            layer = convlstm.ConvLSTM(img_size=img_size_list[i],
                                        input_dim=input_channel, 
                                         hidden_dim=hidden_channels[i],
                                         kernel_size=kernel_size[i],
                                         stride=stride[i],
                                         padding=padding[i],
                                         cnn_dropout=0.2, 
                                         rnn_dropout=0.,
                                         batch_first=True, 
                                         bias=True, 
                                         peephole=False, 
                                         layer_norm=False,
                                         return_sequence=True,
                                         bidirectional=True)
            convlstm_layer.append(layer)  
            input_channel = hidden_channels[i]
    '''
    # gradient check
    layer_num = 1
    convlstm = ConvLSTMNetwork([[64, 64], ], input_channels=128, hidden_channels=[128, 128], ouput_channels=2,
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