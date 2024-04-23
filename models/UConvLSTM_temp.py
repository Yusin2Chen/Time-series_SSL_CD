import torch
from .biconvlstm import ConvLSTM
from torch import nn
from torch.autograd import Variable
from .ResYnet import BasicBlock, ResYnet, ResYnet_


class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=bias, dilation=dilation))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


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
        self.out1 = nn.Sequential(nn.ReLU(inplace=True),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(inplace=True),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        #self.out2 = nn.Sequential(nn.ReLU(),
        #                          conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
        #                          nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
        #                          conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))

    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        contrast = self.out1(input_tensor)
        #consist = self.out2(input_tensor)
        contrast = nn.Sigmoid()(contrast)

        return gfeat, contrast, contrast


class ConvLSTMNetwork_(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetwork_, self).__init__()

        self.enocer = ResYnet_([2, 2, 2, 2], BasicBlock, width=1, in_channel=4)
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
                                    conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(inplace=True),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out1 = nn.Sequential(nn.ReLU(inplace=True),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(inplace=True),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        #self.out2 = nn.Sequential(nn.ReLU(),
        #                          conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
        #                          nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
        #                          conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))

    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        contrast = self.out1(input_tensor)
        #consist = self.out2(input_tensor)
        contrast = nn.Sigmoid()(contrast)

        return gfeat, contrast, contrast


class ConvLSTMNetwork13(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetwork13, self).__init__()

        self.enocer = ResYnet([2, 2, 2, 2], BasicBlock, width=1, in_channel=13)
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
        self.out1 = nn.Sequential(nn.ReLU(inplace=True),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(inplace=True),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        #self.out2 = nn.Sequential(nn.ReLU(),
        #                          conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
        #                          nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
        #                          conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))

    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        contrast = self.out1(input_tensor)
        #consist = self.out2(input_tensor)
        contrast = nn.Sigmoid()(contrast)

        return gfeat, contrast, contrast

class ConvLSTMNetworkC(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetworkC, self).__init__()

        self.enocer = ResYnet_([2, 2, 2, 2], BasicBlock, width=1, in_channel=4)
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
        self.active = nn.Sequential(nn.ReLU(),
                                    conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out0 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        self.out2 = nn.Sequential(conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1),
                                 nn.ReLU(),
                                 conv1x1(int(hidden_channels[-1] * 1), ouput_channels, stride=1))
        self.out1 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1, bias=False),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), 1, stride=1, bias=False))

    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        #input_tensor = self.out1(input_tensor)
        #supfet = self.out2(input_tensor)

        supfet = self.out1(input_tensor)
        supfet = nn.Sigmoid()(supfet)
        #supfet = self.out2(supfet)
        #contst = nn.Sigmoid()(supfet)
        return gfeat, input_tensor, supfet


class MLP:
    def __init__(self):
        self.net = nn.Sequential(nn.ReLU(),
                                 conv3x3(16, 16, stride=1),
                                 nn.BatchNorm2d(16), nn.ReLU(),
                                 conv1x1(16, 2, stride=1))
    def forwar(self, x):
        return self.net(x)


class ConvLSTMNetworkKmeans(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetworkKmeans, self).__init__()

        self.enocer = ResYnet_([2, 2, 2, 2], BasicBlock, width=1, in_channel=4)
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
        self.active = nn.Sequential(nn.ReLU(),
                                    conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out0 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        self.out2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(ouput_channels, ouput_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False))
        self.out1 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))


    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        contrast = self.out1(input_tensor)
        # consist = self.out2(input_tensor)
        contrast = nn.Sigmoid()(contrast)

        return contrast


class ConvLSTMNetworkC13(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetworkC13, self).__init__()

        self.enocer = ResYnet([2, 2, 2, 2], BasicBlock, width=1, in_channel=13)
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
        self.active = nn.Sequential(nn.ReLU(),
                                    conv3x3(hidden_channels[-1] * 4, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out0 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))
        self.out2 = nn.Sequential(nn.Sigmoid(), nn.Conv2d(ouput_channels, ouput_channels, kernel_size=1, stride=1,
                                            padding=0, bias=False))
        self.out1 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), ouput_channels, stride=1))


    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        #input_tensor = self.out1(input_tensor)
        #supfet = self.out2(input_tensor)

        supfet = self.out1(input_tensor)

        contst = nn.Sigmoid()(supfet)
        #contst = self.out2(supfet)

        return gfeat, supfet, contst

class ConvLSTMNetworkCS(torch.nn.Module):
    def __init__(self, img_size_list, input_channels, hidden_channels, ouput_channels, kernel_size, num_layers,
                 bidirectional=False):
        super(ConvLSTMNetworkCS, self).__init__()

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
        self.active = nn.Sequential(nn.ReLU(),
                                    conv3x3(hidden_channels[-1] * 4, int(hidden_channels[-1] * 2), stride=1),
                                    nn.BatchNorm2d(int(hidden_channels[-1] * 2)), nn.ReLU(),
                                    conv1x1(int(hidden_channels[-1] * 2), int(hidden_channels[-1] * 1), stride=1))
        self.out0 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), int(hidden_channels[-1] * 1), stride=1))

        self.out1 = nn.Sequential(nn.ReLU(),
                                  conv3x3(hidden_channels[-1] * 2, int(hidden_channels[-1] * 1), stride=1),
                                  nn.BatchNorm2d(int(hidden_channels[-1] * 1)), nn.ReLU(),
                                  conv1x1(int(hidden_channels[-1] * 1), ouput_channels, stride=1))


    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        input_tensor, gfeat = self.enocer(x1, x2)
        input_tensor = self.active(input_tensor.squeeze())
        input_tensor = input_tensor.unsqueeze(0)
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        input_tensor = input_tensor.view(-1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        #input_tensor = self.out1(input_tensor)
        #supfet = self.out2(input_tensor)

        supfet = self.out1(input_tensor)

        contst = self.out0(input_tensor)
        #contst = nn.Sigmoid()(supfet)

        return gfeat, supfet, contst

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