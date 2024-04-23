"""
DDP training for Contrastive Learning
"""
from __future__ import print_function
import torch
import collections
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from dataset.data_pair import ImgPair
from models.ResUnetStd import ResUnet182
from labelprop.common import LabelPropVOS_CRW
from thresholding import threshold_rosin, threshold_rosin2

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def Rnormalize_S2(imgs):
    for i in range(4):
        imgs[i, :, :] = (imgs[i, :, :] * S2_STD[i]) + S2_MEAN[i]
    return imgs


def parse_option():

    parser = argparse.ArgumentParser('argument for test')
    # specify folder SpaceNet7
    parser.add_argument('--data_folder', type=str, default='/workspace/S2SR', help='path to data')
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # output
    #parser.add_argument('--out_dir', type=str, default='/workspace/S2SR', help='path to save linear classifier')
    parser.add_argument('--out_dir', type=str, default='./result_S2SR_thres', help='path to save linear classifier')
    parser.add_argument('--score', action='store_true', default=True, help='score prediction results using ground-truth data')
    parser.add_argument('--preview_dir', type=str, default='./preview_S2SR_thres', help='path to preview dir (default: no previews)')


    opt = parser.parse_args()


    if not os.path.isdir(opt.preview_dir):
        os.makedirs(opt.preview_dir)

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)

    return opt

def get_train_loader(args):
    # load datasets
    train_set = ImgPair(args.data_folder,
                        datatype='S2',
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        unlabeled=True)

    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader

def encoder_factory(model):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels: the number of output channels
    """
    # load pre-trained model
    print('==> loading pre-trained model')
    pretrained_model = os.path.join('./save_student/resunet', 'student_74_-0.6663037770324283.pth')
    ckpt = torch.load(pretrained_model)
    pretrained_dict = ckpt['target_network_state_dict']
    model_dict = model.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model.cuda()


def make_onehot(mask, cls_num=2):
    # convert mask tensor with probabilities to a one-hot tensor
    b, h, w = mask.shape
    one_hot = torch.zeros((cls_num, h, w))
    one_hot.scatter_(0, mask.long(), 1)
    return one_hot

def mask2tensor(mask, idx, num_classes=2):
    h,w = mask.shape
    mask_t = torch.zeros(1,num_classes,h,w)
    mask_t[0, idx] = mask
    return mask_t

def validate(val_loader, model, labelprop, args):
    """
    evaluation
    """
    # switch to evaluate mode
    model.eval()
    # main validation loop
    with torch.no_grad():
        for idx, (batch) in enumerate(val_loader):

            # unpack sample
            image = batch['image']
            b, n, c, h, w = image.shape
            image = image.view(b * n, c, h, w).contiguous()
            image = torch.nan_to_num(image)
            nb, nc, nh, nw = image.shape
            feats = model(image.cuda(), mode=1)
            feats = feats.view(b, n, -1, nh, nw).contiguous()
            pre_masks = []
            pre_feats = []
            # # cal whole sequence

            # for ascending image pairs
            for i in range(n - 1):
                pred1 = copy.deepcopy(F.normalize(feats[:, i, :, :, :], p=2, dim=1))
                pred_fst = copy.deepcopy(F.normalize(feats[:, -1, :, :, :], p=2, dim=1))
                prediction = -1 * torch.cosine_similarity(pred1, pred_fst, dim=1) + 1
                for_mask = prediction.squeeze().cpu().numpy()
                mask = threshold_rosin2(for_mask)

                # images
                img1 = copy.deepcopy(image[-1, :, :, :])
                img2 = copy.deepcopy(image[i, :, :, :])
                pre_img = Rnormalize_S2(img1.squeeze())
                pre_img = pre_img.cpu().numpy() / 10000
                pos_img = Rnormalize_S2(img2.squeeze())
                pos_img = pos_img.cpu().numpy() / 10000

                # save predictions
                id = batch["fold"][0].split('/')[0] + '_thres_' + str(i)
                # print(mask.shape)
                output_img = Image.fromarray(mask.astype(np.uint8))
                # print(os.path.join(args.out_dir, batch["id"][0]+'/'+id+'.tif'))
                #output_img.save(os.path.join(args.out_dir, batch["fold"][0] + '/' + id + '.tif'))
                output_img.save(os.path.join(args.out_dir, id + '.tif'))

                # save preview
                if args.preview_dir is not None:
                    display_channels = [2, 1, 0]
                    brightness_factor = 3
                    plt.rcParams['figure.dpi'] = 300
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    pre_img = pre_img[display_channels, :, :]
                    pre_img = np.rollaxis(pre_img, 0, 3)
                    pos_img = pos_img[display_channels, :, :]
                    pos_img = np.rollaxis(pos_img, 0, 3)
                    ax1.imshow(np.clip(pre_img * brightness_factor, 0, 1))
                    ax1.set_title("pre")
                    ax1.axis("off")
                    ax2.imshow(np.clip(pos_img * brightness_factor, 0, 1))
                    ax2.set_title("post")
                    ax2.axis("off")
                    ax3.imshow(mask)
                    ax3.set_title("prediction")
                    ax3.axis("off")
                    plt.savefig(os.path.join(args.preview_dir, id), bbox_inches='tight')
                    plt.close()



def main(args):
    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False

    # parameters
    parameters = collections.namedtuple("TEST", "KNN, CXT_SIZE, RADIUS, TEMP")
    par_track = parameters(KNN=1, CXT_SIZE=1, RADIUS=1, TEMP=0.001)
    labelprop = LabelPropVOS_CRW(par_track)
    # build model
    online_network = ResUnet182(width=1, in_channel=4)
    model = encoder_factory(online_network)

    # build dataset
    train_loader = get_train_loader(args)

    # inference
    validate(train_loader, model, labelprop, args)


if __name__ == '__main__':
    args = parse_option()
    main(args)
