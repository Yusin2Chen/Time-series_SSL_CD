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
    parser.add_argument('--model_path', type=str, default='./save_cls_YLSTM_256_crw2', help='path to save model')
    parser.add_argument('--save', type=str, default='./save_cls_YLSTM_256_crw2', help='path to save linear classifier')

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


class CRW_Loss(nn.Module):
    def __init__(self, args, vis=None):
        super(CRW_Loss, self).__init__()
        self.args = args

        self.edgedrop_rate = getattr(args, 'dropout', 0)
        self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.temperature = getattr(args, 'temp', getattr(args, 'temperature', 0.07))

        self.xent = nn.CrossEntropyLoss(reduction="none")
        self._xent_targets = dict()

        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.featdrop = nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.flip = getattr(args, 'flip', False)
        self.sk_targets = getattr(args, 'sk_targets', False)
        self.vis = vis
        self.EPS = 1e-20

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)
        A = torch.einsum('bctn,bctm->btnm', x1, x2)
        return A.squeeze(1) if in_t_dim < 4 else A

    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''
        if zero_diagonal:
            A = self.zeroout_diag(A)
        if do_dropout and self.edgedrop_rate > 0:
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20
        return F.softmax(A / self.temperature, dim=-1)

    def pixels_to_nodes(self, x):
        '''
            pixel maps -> node embeddings
            Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)
            Inputs:
                -- 'x' (B x N x C x T x h x w), batch of images, n=1
            Outputs:
                -- 'feats' (B x C x T x N), node embeddings, n=spatches
                -- 'maps'  (B x N x C x T x H x W), node feature maps, n=spatches
        '''
        B, N, C, T, h, w = x.shape
        maps = x.flatten(0, 1)
        #maps = torch.sum(x, (-2, -1), keepdim=True)
        H, W = maps.shape[-2:]

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        if N == 1:  # flatten single image's feature map to get node feature 'maps'
            maps = maps.permute(0, -2, -1, 1, 2).contiguous()
            maps = maps.view(-1, *maps.shape[3:])[..., None, None]
            N, H, W = maps.shape[0] // B, 1, 1

        # compute node embeddings by spatially pooling node feature maps
        feats = maps.sum(-1).sum(-1) / (H * W)
        feats = feats.transpose(-1, -2).transpose(-1, -2)
        feats = F.normalize(feats, p=2, dim=1)

        feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1)
        maps = maps.view(B, N, *maps.shape[1:])

        return feats, maps

    def forward(self, x, just_feats=False, ):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        #T_o, C_o, H_o, W_o = x.shape
        #x = nn.functional.unfold(x, (8, 8), stride=8)
        #x = x.view(T_o, C_o, 8, 8, -1).contiguous()
        #x = x.permute(4, 1, 0, 2, 3).contiguous()
        #x = x.unsqueeze(0)
        #B, _N, C, T, H, W = x.shape
        #################################################################
        x = x.unsqueeze(0)
        B, T, C, H, W = x.shape
        _N = 1
        # Pixels to Nodes
        x = x.transpose(1, 2).view(B, _N, C, T, H, W)
        #################################################################
        q, mm = self.pixels_to_nodes(x)
        B, C, T, N = q.shape

        if just_feats:
            h, w = np.ceil(np.array(x.shape[-2:]) / self.map_scale).astype(np.int)
            return (q, mm) if _N > 1 else (q, q.view(*q.shape[:-1], h, w))
        #################################################################
        # Compute walks
        #################################################################
        walks = dict()
        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T - 1)]
        #################################################### Palindromes
        if not self.sk_targets:
            A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T - 1)]
            AAs = []
            for i in list(range(1, len(A12s))):
                g = A12s[:i + 1] + A21s[:i + 1][::-1]
                aar = aal = g[0]
                for _a in g[1:]:
                    aar, aal = aar @ _a, _a @ aal
                AAs.append((f"l{i}", aal) if self.flip else (f"r{i}", aar))
            for i, aa in AAs:
                walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]
        #################################################### Sinkhorn-Knopp Target (experimental)
        else:
            a12, at = A12s[0], self.stoch_mat(As[:, 0], do_dropout=False, do_sinkhorn=True)
            for i in range(1, len(A12s)):
                a12 = a12 @ A12s[i]
                at = self.stoch_mat(As[:, i], do_dropout=False, do_sinkhorn=True) @ at
                with torch.no_grad():
                    targets = self.sinkhorn_knopp(at, tol=0.001, max_iter=10, verbose=False).argmax(-1).flatten()
                walks[f"sk {i}"] = [a12, targets]

        #################################################################
        # Compute loss
        #################################################################
        xents = [torch.tensor([0.]).to(self.args.device)]
        #diags = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A + self.EPS).flatten(0, -2)
            loss = self.xent(logits, target).mean()
            #acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            #diags.update({f"{H} xent {name}": loss.detach(), f"{H} acc {name}": acc})
            xents += [loss]

        loss = sum(xents) / max(1, len(xents) - 1)

        return loss

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B, N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def sinkhorn_knopp(self, A, tol=0.01, max_iter=1000, verbose=False):
        _iter = 0
        if A.ndim > 2:
            A = A / A.sum(-1).sum(-1)[:, None, None]
        else:
            A = A / A.sum(-1).sum(-1)[None, None]
        A1 = A2 = A
        while (A2.sum(-2).std() > tol and _iter < max_iter) or _iter == 0:
            A1 = F.normalize(A2, p=1, dim=-2)
            A2 = F.normalize(A1, p=1, dim=-1)
            _iter += 1
            if verbose:
                print(A2.max(), A2.min())
                print('row/col sums', A2.sum(-1).std().item(), A2.sum(-2).std().item())
        if verbose:
            print('------------row/col sums aft', A2.sum(-1).std().item(), A2.sum(-2).std().item())
        return A2


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
        self.crw_loss = CRW_Loss(args)

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
            label_a = label[0]
            image_a = torch.nan_to_num(image_a)
            t_image_a = self.colorize(image_a)
            condn_a = [image_a[-1, :, :, :].unsqueeze(0) for i in range(ni - 1)]
            condn_a = torch.cat(condn_a, dim=0)

        # compute query feature
        g_feata, feat_consisa, feat_contraa = self.online_network(image_a[:-1].cuda(), condn_a.cuda())
        with torch.no_grad():
            g_featat, feat_consisat, feat_contraat = self.target_network(t_image_a[:-1].cuda(), condn_a.cuda())
        # supervised contrastive loss
        _, _, h, w = feat_contraa.shape
        new_h, new_w = 32, 32
        if h > new_h:
            # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            feat_crop = feat_contraa[:, :, top: top + new_h, left: left + new_w]
            #feat_contraat = feat_contraat[:, :, top: top + new_h, left: left + new_w]
            #label_a = label_a[:, top: top + new_h, left: left + new_w]
        else:
            feat_crop = feat_contraa
        #s_list = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
        s_list = random.sample([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8],
                                [7, 8, 9], [8, 9, 10]], 1)
        crw_loss = self.crw_loss(feat_crop[s_list, :, :, :].squeeze())

        feat_1 = torch.cat([F.normalize(feat_contraa.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1),
                           F.normalize(feat_contraat.permute(0, 2, 3, 1).contiguous().view(-1, 16).unsqueeze(1), dim=-1)], dim=1)
        label_1 = label_a + label_a
        label_1 = label_1.view(-1).contiguous()
        # mask the aviable pixels
        msk0 = (label_1 == 0)
        msk1 = (label_1 == 2)
        # print(label1.shape, feat1.shape)
        label0 = label_1[msk0]
        label1 = label_1[msk1]
        feat0 = feat_1[msk0, :, :]
        feat1 = feat_1[msk1, :, :]
        #####################################################
        if len(label0) > len(label1):
            indice = random.sample(range(len(label0)), max(100, label1.shape[0]))
            indice = torch.tensor(indice)
            slabel = torch.cat([label1, label0[indice]], dim=0)
            sfeat = torch.cat([feat1, feat0[indice, :, :]], dim=0)
        else:
            slabel = torch.cat([label1, label0], dim=0)
            sfeat = torch.cat([feat1, feat0], dim=0)
        sindice = random.sample(range(len(slabel)), min(int(len(slabel)), 2000))
        sindice = torch.tensor(sindice)
        ############################################################################################################
        loss1 = 1 * self.sup_con(sfeat[sindice, :], slabel[sindice].cuda().long())
        loss2 = 0.1 * crw_loss

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

