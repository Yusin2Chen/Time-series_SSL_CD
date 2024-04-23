from collections import OrderedDict
import torch
import torch.nn as nn

from torchvision import transforms


from packed_sequence import PackedSequence
from sequence import pad_packed_images


from .augmentation import (
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
    RandomVerticalFlip,
    ColorJitter
)

from .geometric import centralize

from .misc import normalize_min_max, denormalize_min_max


class RandomAugmentation(nn.Module):
    """ Random Augmentation that takes packed sequence and perform sequence of transformations of selected images
    in batch based Bernoulli distribution. The module  translates all padded images to the center of the max-image
    in the input batch and normalize every image based on dataset mean and std. we guarantee a better control in
    photometric and geometric operations.


    """
    def __init__(self, rgb_mean, rgb_std):
        super(RandomAugmentation, self).__init__()

        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self.photometric = nn.Sequential(OrderedDict([
            ('ColorJitter', ColorJitter(p=0.3, brightness=[0.0, 0.3], contrast=[0.7, 1.0], saturation=[0.4, 0.1], hue=[0.0, 0.5])),
            ('Solarize', RandomSolarize(p=0.2, thresholds=[0.95, 1.0], additions=None)),
            ('Posterize', RandomPosterize(p=0.1, bits=[4, 8])),
            ('Sharpness', RandomSharpness(p=0.1, sharpness=[0.0, 1.0])),
            ('Equalize', RandomEqualize(p=0.1)),
            ('Grayscale', RandomGrayscale(p=0.1))
        ]))

        self.geometric = nn.Sequential(OrderedDict([
            ('horizontal_flip', RandomHorizontalFlip(p=0.2)),
            ('VerticalFlip', RandomVerticalFlip(p=0.2)),
            ('Rotation', RandomRotation(p=0.2, theta=60)),
            ('Perspective', RandomPerspective(p=0.3, distortion_scale=0.3)),
            ('Affine', RandomAffine(p=0.3, theta=30, h_trans=0.0, v_trans=0.0, scale=[0.8, 1.6], shear=[0.0, 0.2]))
        ]
        ))

        self.filter = nn.Sequential(OrderedDict([
            ('MotionBlur', RandomMotionBlur(p=0.1, kernel_size=[5, 11], angle=[0, 90], direction=[-1, 1]))
        ]))

    def show(self, inp, out):

        for (inp_i, out_i) in zip(inp, out):
            inp_i = transforms.ToPILImage()(inp_i.cpu()).convert("RGB")
            out_i = transforms.ToPILImage()(out_i.cpu()).convert("RGB")

            inp_i.show()
            out_i.show()

    def _unpack_input(self, inp):
        """
            PackedSequence List(C W H )--> Tensors B C W H
        """
        if isinstance(inp, PackedSequence):

            # Pad the input images
            inp, valid_size = pad_packed_images(inp)
            return inp, valid_size
        else:
            raise TypeError("Input type is not a PackedSequence Got {}".format(type(input)))

    def _normalize_image(self, img):
        """
            zero out again the padded region, and discard the effect of photometric and filtering operations out of
            valid size.
        """
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    def _msk(self, x, bbx):
        """
            zero out again the padded region, and discard the effect of photometric and filtering operations out of
            valid size.
        """

        msk = torch.zeros_like(x)
        for msk_i, bbx_i in zip(msk, bbx):

            kernel = torch.ones(int(bbx_i[2]), int(bbx_i[3]))
            msk_i[:, int(bbx_i[0]): int(bbx_i[0]+bbx_i[2]), int(bbx_i[1]): int(bbx_i[1]+bbx_i[3])] = kernel

        return x * msk

    def forward(self, x, masking=True):
        # Unpack inputs
        x, valid_size = self._unpack_input(x)

        # translate pure image to center
        x, bbx = centralize(x, valid_size)

        # normalize min_max !better photometric augmentation control
        x, x_min, x_max = normalize_min_max(x)

        # run augmentation
        x = self.geometric(x)
        x = self.photometric(x)
        x = self.filter(x)

        # denormalize min_max
        x = denormalize_min_max(x, x_max=x_max, x_min=x_min)

        x = self._msk(x, bbx)

        # normalize based on mean and std
        x = self._normalize_image(x)

        # Pack output
        x = [item for item in x]
        return PackedSequence(x)
