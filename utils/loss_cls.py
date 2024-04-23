import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.percentile import Percentile

class HardNegtive_loss(torch.nn.Module):

    def __init__(self, tau_plus=0.1, beta=1.0, temperature=0.5, alpha=256, estimator='hard', normalize=True):
        super(HardNegtive_loss, self).__init__()
        self.tau_plus = tau_plus
        self.beta = beta
        self.normalize = normalize
        self.temperature = temperature
        self.estimator = estimator
        self.alpha = alpha

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        if self.normalize:
            out_1 = F.normalize(out_1, p=2, dim=1)
            out_2 = F.normalize(out_2, p=2, dim=1)

        batch_size, c = out_1.shape
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        # eqco
        #print(batch_size, Ng.shape)
        #loss = (- torch.log(pos / (pos + self.alpha / Ng.shape[0] * Ng))).mean()

        return loss

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])

class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.inter_cal = None
        self.intra_cal = None
        self.neg_cal = None
        self.neg_inter_shift = 0.31361241889448443
        self.pos_inter_shift = 0.1754346515479633
        self.pos_intra_shift = 0.45828472207
        self.pos_inter_weight = 1
        self.pos_intra_weight = 1
        self.neg_inter_weight = 1
        self.pointwise = False
        self.stabalize = True
        self.feature_samples = 11
        self.neg_samples = 5

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, cd, similar=True):
        # noting 此函数的意思是一个像素的值在正负样本中的值都相同，在正中希望cd值小（距离近）则减去大数，(fd - shift)为负，让loss为正
        # 在负样本对中希望cd值大（距离远）则减去小数，(fd - shift)为正，让loss为负
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            # 这里反过来了啊，越不相似值越大！！！！！！！！！！距离越远，全按照欧氏距离思路
            fd = -1 * torch.cosine_similarity(f1, f2, dim=1) + 1

            if self.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([-2, -1], keepdim=True)
                fd = fd - fd.mean() + old_mean

        if similar:
            fv = torch.clone(fd).flatten()
            fv = fv[fv.argsort()]
            shift = Percentile()(fv, [70])
            del fv
            #shift = 0.46
        else:
            fv = torch.clone(fd).flatten()
            fv = fv[fv.argsort()]
            shift = Percentile()(fv, [5])
            del fv
            #shift = 0.32

        return cd, fd - shift

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor, orig_code: torch.Tensor,
                nega_feats: torch.Tensor, nega_feats_pos: torch.Tensor, nega_code: torch.Tensor,
                ):
        # code was set as the change map of one pair of images
        #coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]
        #coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        #coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = orig_feats #sample(orig_feats, coords1)
        feats_pos = orig_feats_pos #sample(orig_feats_pos, coords1)
        code = orig_code #sample(orig_code, coords1)

        pos_inter_loss, pos_c = self.helper(
            feats, feats_pos, code, similar=True)
        pos_inter_loss = -1 * pos_inter_loss[pos_c > 0].mean() + 1 * pos_inter_loss[pos_c < 0].mean()

        # negative loss term
        feats_neg = nega_feats #sample(nega_feats, coords2)
        feats_neg_pos = nega_feats_pos #sample(nega_feats_pos, coords2)
        code_neg = nega_code #sample(nega_code, coords2)
        neg_inter_loss, neg_c = self.helper(
                feats_neg, feats_neg_pos, code_neg, similar=False)
        neg_inter_loss = -1 * neg_inter_loss[neg_c > 0].mean() + 1 * neg_inter_loss[neg_c < 0].mean()

        return self.pos_inter_weight * pos_inter_loss + self.neg_inter_weight * neg_inter_loss
