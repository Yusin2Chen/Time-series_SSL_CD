import torch
from .biconvlstm import ConvLSTM
from torch import nn
from torch.autograd import Variable
from .ResYnet import BasicBlock, ResYnet
from torch import einsum
from torch.nn import init
from einops import rearrange
import torch.nn.functional as F


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

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ConvLSTMNetwork(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetwork, self).__init__()

        self.enocer = ResYnet([2, 2, 2, 2], BasicBlock, width=1, in_channel=4)
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
        self.active = nn.Sequential(nn.ReLU(inplace=True),
                                    conv3x3(hidden_channels[-1] * 4, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(inplace=True),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out = nn.Sequential(nn.ReLU(inplace=True),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(inplace=True),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        self.embed1 = nn.Embedding(1, int(hidden_channels[-1] * 1))
        self.embed2 = nn.Embedding(1, int(hidden_channels[-1] * 1))
        self.norm = Normalize(2)

    def forward(self, x1, x2, mode=0):
        if mode == 0:
            # print(x1.shape, x2.shape)
            input_tensor, gfeat = self.enocer(x1, x2)
            input_tensor = self.active(input_tensor.squeeze())
            input_tensor = input_tensor.unsqueeze(0)
            for i in range(self.num_layers):
                input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
            input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
            feat = self.out(input_tensor)
            b, c, w, h = feat.shape
            feat1 = feat
            ins1 = self.embed1.weight.expand(b, c).unsqueeze(dim=2)
            ins2 = self.embed2.weight.expand(b, c).unsqueeze(dim=2)
            embd = torch.cat([ins1, ins2], dim=-1)
            divs = torch.cat([self.embed1.weight, self.embed2.weight], dim=0)
            loss_dv = self.uniform_loss(self.norm(divs))
            feat = rearrange(feat, 'b c h w -> b h w c')
            fb, fm, fn, fc = feat.shape
            # Flatten feat
            feat = feat.contiguous().view(fb, -1, fc)
            proj = einsum('ijk,ikl -> ijl', feat, embd)
            soft_one_hot = F.gumbel_softmax(proj, tau=0.8, dim=-1, hard=False)
            # diversity loss + kl divergence to the prior loss
            qy = F.softmax(proj, dim=1)
            diff = 5e-4 * torch.sum(qy * torch.log(qy * 6 + 1e-10), dim=1).mean()
            # pixel-wise feature
            embd = rearrange(embd, 'i k l -> i l k')
            feat2 = einsum('ijl, ilk -> ijk', soft_one_hot, embd)
            feat2 = feat2.contiguous().view(fb, fm, fn, fc)
            feat2 = rearrange(feat2, 'b h w c -> b c h w')
            # normalization
            feat1 = self.norm(feat1)
            feat2 = self.norm(feat2)

            return gfeat, feat1, feat2, diff + loss_dv
        else:
            input_tensor, gfeat = self.enocer(x1, x2)
            input_tensor = self.active(input_tensor.squeeze())
            input_tensor = input_tensor.unsqueeze(0)
            for i in range(self.num_layers):
                input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
            input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
            feat = self.out(input_tensor)
            b, c, w, h = feat.shape
            feat1 = feat
            ins1 = self.embed1.weight.expand(b, c).unsqueeze(dim=2)
            ins2 = self.embed2.weight.expand(b, c).unsqueeze(dim=2)
            embd = torch.cat([ins1, ins2], dim=-1)
            feat = rearrange(feat, 'b c h w -> b h w c')
            fb, fm, fn, fc = feat.shape
            # Flatten feat
            feat = feat.contiguous().view(fb, -1, fc)
            proj = einsum('ijk,ikl -> ijl', feat, embd)
            soft_one_hot = F.gumbel_softmax(proj, tau=0.8, dim=-1, hard=False)
            soft_one_hot = soft_one_hot.contiguous().view(fb, fm, fn, -1)
            soft_one_hot = rearrange(soft_one_hot, 'b h w c -> b c h w')
            #soft_one_hot = torch.argmax(soft_one_hot, dim=1)
            return None, None, None, soft_one_hot

    def align_loss(sef, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

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