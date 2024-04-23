import os
import copy
import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from models.UConvLSTM_temp import ConvLSTMNetworkC
from torch.utils.data import DataLoader
from dataset.data_random_lstm import MTCD
from utils.loss_cls_pix import ContrastiveCorrelationLoss
from utils.supcontrast import SupConLoss
# augmentation
from dataset.augmentation.augmentation import (
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPosterize,
    RandomSharpness,
    RandomEqualize,
    RandomRotation,
    RandomSolarize,
    RandomAffine,
    RandomPerspective,
    RandomMotionBlur,
    RandomGaussianBlur,
    RandomVerticalFlip,
    ColorJitter
)

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    # 1600
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # resume path
    parser.add_argument('--resume', action='store_true', default=False, help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='ResUnet182', choices=['CMC_mlp3614', 'alexnet', 'resnet'])
    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # add new views
    #/workspace/S2_MTCD #'/workspace/MTCDE'
    parser.add_argument('--data_dir_train', type=str, default='/workspace/S2SR', help='path to training dataset')
    parser.add_argument('--model_path', type=str, default='./save_cls_YLSTM_256_sigmoid', help='path to save model')
    parser.add_argument('--save', type=str, default='./save_cls_YLSTM_256_sigmoid', help='path to save linear classifier')

    opt = parser.parse_args()

    # set up saving name
    opt.save_path = os.path.join(opt.save, opt.model)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    if (opt.data_dir_train is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))

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
                        transform=True,
                        crop_size=args.crop_size)
    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    return train_loader


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class BYOLTrainer:
    def __init__(self, args, online_network, target_network, criterion, optimizer, device):
        self.online_network = online_network
        self.target_network = target_network
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.sup_con = SupConLoss()
        self.sup_loss2 = nn.CrossEntropyLoss(weight=torch.tensor([1., 8.])).cuda()
        ## 随机从 0 ~ 2 之间亮度变化，1 表示原图 | # 随机从 0 ~ 2 之间对比度变化，1 表示原图
        #self.sup_loss2 = FocalLoss(weight=torch.tensor([1., 5.])).cuda()
        self.colorize = ColorJitter(p=1.0, brightness=[0.85, 1.15], contrast=[0.85, 1.15], saturation=None, hue=None)
        self.loss = nn.BCELoss(reduction='none')

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader):
        tasks = ['contrast', 'consist']
        niter = 0
        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, batch in enumerate(train_loader):
                image = batch['image']
                label = batch['label']
                loss = self.update(image, label)

                # total loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
            # save checkpoints
            if (epoch_counter + 1) % 50 == 0:
                self.save_model(os.path.join(self.savepath, 'BOYL_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))
            torch.cuda.empty_cache()

    def update(self, image, label):
        with torch.no_grad():
            bi, ni, ci, hi, wi = image.shape
            image_a = image[0]
            label_a = label[0].float()
            image_a = torch.nan_to_num(image_a)
            t_image_a = self.colorize(image_a)
            condn_a = [image_a[-1, :, :, :].unsqueeze(0) for i in range(ni - 1)]
            condn_a = torch.cat(condn_a, dim=0)
        # compute query feature
        g_feata, feat_consisa, sup_mapa = self.online_network(image_a[:-1].cuda(), condn_a.cuda())

        y_train_mean = label_a.mean()
        prediction_mean = sup_mapa.view(-1).cpu().detach().round().mean()
        bi_cls_w2 = 1 / (1 - y_train_mean)
        bi_cls_w1 = 1 / y_train_mean - bi_cls_w2
        target = label_a.view(-1).cuda()
        if prediction_mean > 1.5 * y_train_mean or y_train_mean > 0.4:
            loss = self.loss(sup_mapa.view(-1), target)
        else:
            loss = (bi_cls_w1 * target + bi_cls_w2) * self.loss(sup_mapa.view(-1), target)
        return loss.mean()


    def randcat(self, x1, x2):
        return torch.abs(x1 - x2).unsqueeze(0)

    def save_model(self, PATH):
        print('==> Saving...')
        state = {
            'online_network_state_dict': self.online_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, PATH)
        # help release GPU memory
        del state

    def sup_con_gt(self, label):
        c, h, w = label.shape
        new_label = torch.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if torch.all(label[:, i, j]):
                    new_label[i, j] = label[0, i, j]
                else:
                    new_label[i, j] = 255
        return new_label




def main():

    # parse the args
    args = parse_option()
    # set flags for GPU processing if available
    args.device = 'cuda'
    # set the data loader
    train_loader = get_train_loader(args)
    # set the model
    online_network = ConvLSTMNetworkC([[256, 256], ], input_channels=16, ouput_channels=2, hidden_channels=[16, 16],
                                     kernel_size=[[3, 3], ], num_layers=1, bidirectional=True).to(args.device)
    target_network = copy.deepcopy(online_network)
    # cretiration
    criterion = ContrastiveCorrelationLoss()
    # predictor network
    optimizer = torch.optim.Adam(online_network.parameters(), lr=3e-5)
    trainer = BYOLTrainer(args,
                          online_network=online_network,
                          target_network=target_network,
                          criterion=criterion,
                          optimizer=optimizer,
                          device=args.device)
    trainer.train(train_loader)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

