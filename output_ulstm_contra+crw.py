import os
import copy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
from models.UConvLSTM_temp import ConvLSTMNetworkC
from dataset.data_test_CD import MTCD
from crf import dense_crf
# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def Rnormalize_S2(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i,:,:] * S2_STD[i]) + S2_MEAN[i]
    return imgs

def parse_option():

    parser = argparse.ArgumentParser('argument for test')
    # specify folder # '/workspace/S2_MTCD' #'/workspace/MTCDE'
    parser.add_argument('--data_dir_train', type=str, default='/workspace/S2SR', help='path to data')
    parser.add_argument('--model_path', type=str, default='./save_cls_LSTM', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # output
    parser.add_argument('--out_dir', type=str, default='./output_contra+crw_pos_nomask_adjcent50', help='path to save linear classifier')


    opt = parser.parse_args()

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)

    return opt


def get_train_loader(args):
    # load datasets
    train_set = MTCD(args.data_dir_train,
                        datatype="S2",
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        unlabeled=False,
                        transform=False,
                        crop_size=args.crop_size)
    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)
    return train_loader


def lstm_factory(model):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels: the number of output channels
    """
    # load pre-trained model
    print('==> loading pre-trained model')
    pretrained_model = os.path.join('./save_cls_YLSTM_256_mlp_crw_pos_nomask_adjcent/ResUnet182/BOYL_epoch_49_0.47750193645825256.pth')
    ckpt = torch.load(pretrained_model)
    # loading model
    pretrained_dict = ckpt['online_network_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model.cuda()


def validate(val_loader, pretrain, convlstm, args):
    """
    evaluation
    """
    # switch to evaluate mode
    convlstm.eval()

    with torch.no_grad():
        for idx, (batch) in enumerate(val_loader):

            args.score = False
            # unpack sample
            img_asc = batch['image']
            lbl_asc = batch['label']

            with torch.no_grad():
                bi, ni, ci, hi, wi = img_asc.shape
                img_asc = img_asc.view(bi * ni, ci, hi, wi).contiguous()
                img_asc = torch.nan_to_num(img_asc)
                condn_a = [img_asc[-1, :, :, :].unsqueeze(0) for i in range(ni - 1)]
                condn_a = torch.cat(condn_a, dim=0)
            # predict mask
            _, output_tensor, cls_maps = convlstm(img_asc[:-1].cuda(), condn_a.cuda())
            length = cls_maps.shape[0]
            # loop for output
            for i in range(length):
                id = batch["id"][0].split('/')[0] + 'cls_' + str(i)
                np.savez(os.path.join(args.out_dir, id), output_tensor[i, :, :, :].cpu().detach())



def main():

    # parse the args
    args = parse_option()
    # set flags for GPU processing if available
    args.device = 'cuda'
    # set the data loader
    val_loader = get_train_loader(args)
    # set the model
    convlstm= ConvLSTMNetworkC([[256, 256], ], input_channels=16, ouput_channels=2, hidden_channels=[16, 16],
                                     kernel_size=[[3, 3], ], num_layers=1, bidirectional=True).to(args.device)
    convlstm = lstm_factory(convlstm)
    # build model
    pretrain_network = None
    # test it
    validate(val_loader, pretrain_network, convlstm, args)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

