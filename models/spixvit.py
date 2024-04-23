import math
import torch
import torch.nn as nn
import sparseconvnet as scn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbedding(nn.Module):
    r"""PatchEmdedding class
    Args:
        image_size(int): size of the image. assume that image shape is square
        in_channels(int): input channel of the image, 3 for RGB color channel
        embed_size(int): output channel size. This is the latent vector size.
                         and is constant throughout the transformer
        patch_size(int): size of the patch

    Attributes:
        n_patches(int): calculate the number of patches.
        patcher: convert image into patches. Basically a convolution layer with
                 kernel size and stride as of the patch size
    """

    def __init__(self, image_size=224, in_channels=3, embed_size=768, patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.patcher = nn.Conv2d(in_channels, embed_size, patch_size, patch_size)

    def forward(self, x):
        # convert the images into patches
        out = self.patcher(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        return out

class SpixEmbedding(nn.Module):
    r"""PatchEmdedding class
    Args:
        image_size(int): size of the image. assume that image shape is square
        in_channels(int): input channel of the image, 3 for RGB color channel
        embed_size(int): output channel size. This is the latent vector size.
                         and is constant throughout the transformer
        patch_size(int): size of the patch

    Attributes:
        n_patches(int): calculate the number of patches.
        patcher: convert image into patches. Basically a convolution layer with
                 kernel size and stride as of the patch size
    """

    def __init__(self, image_size=256, in_channels=3, embed_size=192, patch_size=16):
        super(SpixEmbedding, self).__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.patcher = scn.Sequential().add(scn.Convolution(2, in_channels, embed_size, 16, 16, False)).add(
            scn.SparseToDense(2, embed_size))
        self.outSpatialSize = self.patcher.input_spatial_size(torch.LongTensor([1, 1]))
        self.input_layer = scn.InputLayer(2, self.outSpatialSize)
        #self.patcher = nn.Conv2d(in_channels, embed_size, kernel_size=16, stride=16)

    def unfold_patch(self, x, seg, crop):
        b, c, h, w = x.shape
        ori_img_size = 256
        psz = 16
        edg = ori_img_size - psz
        # unfolder x to B * patches
        img_list = []
        seg_list = []
        p = crop.shape[-1]
        for i in range(b):
            for j in range(p):
                seg_ij = (seg[i] == crop[i, j])
                m = seg_ij.nonzero()  # locate the non-zero
                r_min, c_min = m[:, 0].min(), m[:, 1].min()
                r_max, c_max = m[:, 0].max(), m[:, 1].max()
                h = r_max + 1 - r_min
                w = c_max + 1 - c_min
                if h == 0 or w == 0:
                    continue
                if (r_max + 1 - r_min <= psz) and (c_max + 1 - c_min <= psz):
                    # print('all', seg_ij.sum())
                    if r_min > edg and c_min < edg:
                        a = r_min + psz - ori_img_size
                        s_img = x[i, :, r_min:r_min + psz, c_min:c_min + psz]
                        s_seg = seg_ij[r_min:r_min + psz, c_min:c_min + psz].unsqueeze(0)
                        s_img = nn.ZeroPad2d((0, 0, 0, a))(s_img)
                        s_seg = nn.ZeroPad2d((0, 0, 0, a))(s_seg)
                        # print('small 1', s_seg.sum())
                        img_list.append(s_img.unsqueeze(0))
                        seg_list.append(s_seg)
                    elif r_min < edg and c_min > edg:
                        b = c_min + psz - ori_img_size
                        s_img = x[i, :, r_min:r_min + psz, c_min:c_min + psz]
                        s_seg = seg_ij[r_min:r_min + psz, c_min:c_min + psz].unsqueeze(0)
                        s_img = nn.ZeroPad2d((0, b, 0, 0))(s_img)
                        s_seg = nn.ZeroPad2d((0, b, 0, 0))(s_seg)
                        # print('small 2', s_seg.sum())
                        img_list.append(s_img.unsqueeze(0))
                        seg_list.append(s_seg)
                    elif r_min >= edg and c_min >= edg:
                        a = r_min + psz - ori_img_size
                        b = c_min + psz - ori_img_size
                        s_img = x[i, :, r_min:r_min + psz, c_min:c_min + psz]
                        s_seg = seg_ij[r_min:r_min + psz, c_min:c_min + psz].unsqueeze(0)
                        s_img = nn.ZeroPad2d((0, b, 0, a))(s_img)
                        s_seg = nn.ZeroPad2d((0, b, 0, a))(s_seg)
                        # print('small 3', s_seg.sum())
                        img_list.append(s_img.unsqueeze(0))
                        seg_list.append(s_seg)
                    else:
                        s_img = x[i, :, r_min:r_min + psz, c_min:c_min + psz]
                        s_seg = seg_ij[r_min:r_min + psz, c_min:c_min + psz]
                        # print('small 4', s_seg.sum())
                        img_list.append(s_img.unsqueeze(0))
                        seg_list.append(s_seg.unsqueeze(0))
                else:
                    # for calcculation be float
                    s_img = x[i, :, r_min:r_max + 1, c_min:c_max + 1]
                    s_seg = seg_ij[r_min:r_max + 1, c_min:c_max + 1]
                    s_seg = s_seg.to(torch.float32)
                    s_img = F.interpolate(s_img.unsqueeze(0), size=(psz, psz), mode='bilinear', align_corners=False)
                    s_seg = F.interpolate(s_seg.unsqueeze(0).unsqueeze(0), size=(psz, psz), mode='bilinear',
                                          align_corners=False)  # .floor().int()
                    img_list.append(s_img)
                    seg_list.append(s_seg.squeeze(1))

        return img_list, seg_list

    def forward(self, x, seg_in, crop):
        # unfolder x to B * patches
        bz, _, _, _ = x.shape
        img, seg = self.unfold_patch(x, seg_in, crop)
        img = torch.cat(img).permute(0, 2, 3, 1)
        seg = torch.cat(seg)
        #avg = torch.reciprocal(torch.sum(seg.clone(), (1, 2), keepdim=True))
        # print(img.shape, seg.shape)
        nz_idx = seg.nonzero(as_tuple=True)
        nz_img = img[nz_idx]
        del img, seg
        nz_seg = torch.cat([nz_idx[1].unsqueeze(1), nz_idx[2].unsqueeze(1), nz_idx[0].unsqueeze(1)], dim=1)
        # print(nz_img.shape, nz_seg.shape)
        # print(img.shape, seg.shape)
        # convert the images into patches
        out = self.patcher(self.input_layer([nz_seg.long(), nz_img])).squeeze(3)
        #out = out * avg
        #out = self.patcher(img)
        out = out.view(bz, -1, self.embed_size)
        # print(out.shape)
        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=4, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = self.patch_emb = SpixEmbedding(img_size[0], in_chans, embed_dim, patch_size)
        #self.patch_embed = self.patch_emb = PatchEmbedding(img_size[0], in_chans, embed_dim, patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_position_emb(self, inp_emb):
        _, n_patches, embed_size = inp_emb.shape
        position_emb = nn.Parameter(torch.zeros(1, n_patches, embed_size)).to(inp_emb.device)
        return position_emb

    def prepare_tokens(self, x, seg, crop):
        patch_emb = self.patch_emb(x, seg, crop)
        #patch_emb = self.patch_emb(x)
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        inp_emb = torch.cat((cls_token, patch_emb), dim=1)
        inp_emb += self.add_position_emb(inp_emb)
        inp_emb = self.dropout(inp_emb)
        return inp_emb

    def forward(self, x, seg, crop):
        inp_emb = self.prepare_tokens(x, seg, crop)
        for block in self.blocks:
            inp_emb = block(inp_emb)
        inp_emb = self.layer_norm(inp_emb)
        # Fetch only the embedding for class
        final_cls_token = inp_emb[:, 0]
        out = self.head(final_cls_token)
        return out

    def get_dense_feature(self, x, seg, crop):
        inp_emb = self.prepare_tokens(x, seg, crop)
        for block in self.blocks:
            inp_emb = block(inp_emb)
        inp_emb = self.layer_norm(inp_emb)
        return inp_emb

    def get_last_selfattention(self, x, seg, crop):
        x = self.prepare_tokens(x, seg, crop)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, seg, crop, n=1):
        x = self.prepare_tokens(x, seg, crop)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def spvit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def spvit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def spvit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

