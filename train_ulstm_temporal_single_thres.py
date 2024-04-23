import os
import copy
import torch
import random
import argparse
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from models.UConvLSTM_temp import ConvLSTMNetwork
from torch.utils.data import DataLoader
#from dataset.data_random_lstm import MTCD
from dataset.data_random_lstm_thres import MTCD
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
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

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
    parser.add_argument('--model_path', type=str, default='./save_cls_YLSTM_256_thres1', help='path to save model')
    parser.add_argument('--save', type=str, default='./save_cls_YLSTM_256_thres1', help='path to save linear classifier')

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


def SPC(sup_mapa, sup_mapat, pred_a, idx, corr_idx1, crop_size):  # Segmentation Perceptual Consistency:
    _, _, h, w = sup_mapa.shape
    new_h, new_w = crop_size
    if h > new_h:
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        sup_mapa = sup_mapa[:, :, top: top + new_h, left: left + new_w]
        sup_mapat = sup_mapat[:, :, top: top + new_h, left: left + new_w]
        pred_a = pred_a[:, top: top + new_h, left: left + new_w]
        #pred_a = pred_a[:, :, top: top + new_h, left: left + new_w]
    # consitence loss
    pFeature_A = sup_mapa[idx].contiguous().view(16, -1)
    pFeature_B = sup_mapat[corr_idx1].contiguous().view(16, -1)
    seg_A = get_one_hot(pred_a[idx].long()).permute(2, 0, 1).contiguous().view(2, -1)
    seg_B = get_one_hot(pred_a[corr_idx1].long()).permute(2, 0, 1).contiguous().view(2, -1)
    #seg_A = pred_a[idx].contiguous().view(2, -1)
    #seg_B = pred_a[corr_idx1].contiguous().view(2, -1)
    # pFeature_A : perceptual feature map for image A, shape =[c,w*h]
    # pFeature_A : perceptual feature map for image B, shape =[c,w*h]
    # seg_A : one - hot segmentation map for image A, shape =[ n_cls ,w*h]
    # seg_B : one - hot segmentation map for image B, shape =[ n_cls ,w*h]
    # normalize features in A to unit length :
    pFeature_A = nn.functional.normalize(pFeature_A, p=2, dim=0)
    # normalize features in B to unit length :
    pFeature_B = nn.functional.normalize(pFeature_B, p=2, dim=0)
    # prepare correct tensor shapes for computing correlation matrix :
    pFeature_A = pFeature_A.transpose_(1, 0)
    seg_A = seg_A.transpose_(1, 0)
    # compute correlation between perceptual features of two images
    correlation = torch.matmul(pFeature_A, pFeature_B)
    # find optimal matching of perceptual features w/o segmentation constraint (Eq .1 in the paper ):
    max0_no_constraint = torch.max(correlation, dim=1)
    max1_no_constraint = torch.max(correlation, dim=0)
    # find optimal matching of perceptual features under segmentation constraint (Eq .2 in the paper ):
    correlationSeg = torch.matmul(seg_A.float(), seg_B.float())
    correlationSeg = correlation * correlationSeg
    max0_with_constraint = torch.max(correlationSeg, dim=1)
    max1_with_constraint = torch.max(correlationSeg, dim=0)

    # compute the averages to be used in Eq .3:
    mm0_avg = torch.mean(correlation, dim=1)
    mm1_avg = torch.mean(correlation, dim=0)

    # compute perceptual consistency (Eq .3 in the paper ):
    pcA_map = (max0_with_constraint[0] - mm0_avg) / (max0_no_constraint[0] - mm0_avg)
    pcB_map = (max1_with_constraint[0] - mm1_avg) / (max1_no_constraint[0] - mm1_avg)
    pcA_imageLevel = pcA_map.mean()
    pcB_imageLevel = pcB_map.mean()
    pc_overall = min(pcA_imageLevel, pcB_imageLevel)
    # pdb.set_trace()
    return pc_overall, pcA_map, pcB_map

def get_one_hot(label, n=2):
    size = list(label.size())
    label = label.contiguous().view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(n).cuda()
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(n)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

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
            image_b = image[1]
            label_a = label[0]
            label_b = label[1]
            image_a = torch.nan_to_num(image_a)
            image_b = torch.nan_to_num(image_b)
            t_image_a = self.colorize(image_a)
            t_image_b = self.colorize(image_b)
            condn_a = [image_a[-1, :, :, :].unsqueeze(0) for i in range(ni - 1)]
            condn_a = torch.cat(condn_a, dim=0)
            condn_b = [image_b[-1, :, :, :].unsqueeze(0) for i in range(ni - 1)]
            condn_b = torch.cat(condn_b, dim=0)

        # compute query feature
        g_feata, feat_consisa, feat_contraa = self.online_network(image_a[:-1].cuda(), condn_a.cuda())
        g_featb, feat_consisb, feat_contrab = self.online_network(image_b[:-1].cuda(), condn_b.cuda())
        with torch.no_grad():
            g_featat, feat_consisat, feat_contraat = self.target_network(t_image_a[:-1].cuda(), condn_a.cuda())
            g_featbt, feat_consisbt, feat_contrabt = self.target_network(t_image_b[:-1].cuda(), condn_b.cuda())
            # supervised contrastive loss between two images
            # decide which one and it's cloest one
            #g_feat1 = F.normalize(g_feata.squeeze().detach().cpu(), dim=1)
            #g_feat2 = F.normalize(g_featb.squeeze().detach().cpu(), dim=1)
            idx = random.sample(range(ni - 1), 2)
            #anchor_feat1 = g_feat1[idx].unsqueeze(0)
            #anchor_feat2 = g_feat2[idx].unsqueeze(0)
            #pairwise_sims1 = torch.einsum("nf,mf->nm", anchor_feat1, g_feat1)
            #pairwise_sims2 = torch.einsum("nf,mf->nm", anchor_feat2, g_feat2)
            #_, corr_idx1 = pairwise_sims1.topk(2, dim=1, largest=True, sorted=True)
            #corr_idx1 = corr_idx1[0][1]
            #_, corr_idx2 = pairwise_sims2.topk(2, dim=1, largest=True, sorted=True)
            #corr_idx2 = corr_idx2[0][1]
        # consistent loss
        loss_consistent1, _, _ = SPC(torch.clone(feat_consisa), torch.clone(feat_consisat),
                                     torch.clone(label_a).cuda(), idx[0], idx[1], (64, 64))
        loss_consistent2, _, _ = SPC(torch.clone(feat_consisb), torch.clone(feat_consisbt),
                                     torch.clone(label_b).cuda(), idx[0], idx[1], (64, 64))
        # supervised contrastive loss
        feat1 = torch.cat([F.normalize(feat_contraa.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1),
                           F.normalize(feat_contraat.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1)],
                          dim=1)
        feat2 = torch.cat([F.normalize(feat_consisb.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1),
                           F.normalize(feat_consisbt.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1)],
                          dim=1)
        label1 = label_a + label_a
        label2 = label_b + label_b
        label1 = label1.view(-1).contiguous()
        label2 = label2.view(-1).contiguous()
        feat = torch.cat((feat1, feat2), dim=0)
        label = torch.cat((label1, label2), dim=0)
        # mask the aviable pixels
        msk0 = (label == 0)
        msk1 = (label == 2)
        if msk0.any() or msk1.any():
            # print(label1.shape, feat1.shape)
            label0 = label[msk0]
            label1 = label[msk1]
            feat0 = feat[msk0, :, :]
            feat1 = feat[msk1, :, :]
            #####################################################
            if len(label0) > len(label1):
                indice = random.sample(range(len(label0)), max(100, label1.shape[0]))
                indice = torch.tensor(indice)
                slabel = torch.cat([label1, label0[indice]], dim=0)
                sfeat = torch.cat([feat1, feat0[indice, :, :]], dim=0)
            else:
                slabel = torch.cat([label1, label0], dim=0)
                sfeat = torch.cat([feat1, feat0], dim=0)
            sindice = random.sample(range(len(slabel)), min(int(len(slabel)), 1000))
            sindice = torch.tensor(sindice)
            ############################################################################################################
            loss1 = 1 * self.sup_con(sfeat[sindice, :], slabel[sindice].cuda().long())
            loss2 = 0.01 * (2 - loss_consistent1 - loss_consistent2)
        else:
            loss1 = 0.001
            loss2 = 0.01 * (2 - loss_consistent1 - loss_consistent2)
        #print(loss1, loss2)
        return loss1 + loss2


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
    online_network = ConvLSTMNetwork([[256, 256], ], input_channels=16, ouput_channels=2, hidden_channels=[16, 16],
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

