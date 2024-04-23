
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .geometric import (
    hflip,
    vflip,
    rotate,
    warp_perspective,
    get_perspective_transform,
    warp_affine,
    get_affine_matrix2d)

from .photometric import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    solarize,
    equalize,
    posterize,
    sharpness,
    grayscale
)

from .filters import (
    motion_blur,
    gaussian_blur2d
)

from .misc import _adapted_uniform


class AugmentationBase(nn.Module):

    def __init__(self, p=0.2):
        super(AugmentationBase, self).__init__()
        self.p = p

    def apply(self):
        raise NotImplemented

    def generator(self):
        raise NotImplemented

    def _adapted_sampling(self, shape, device, dtype):
        r"""The uniform sampling function that accepts 'same_on_batch'.
        If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
        By default, same_on_batch is set to False.
        """
        _bernoulli = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))
        target = _bernoulli.sample((shape,)).bool()
        return target

    def forward(self, img):

        img_size = img.shape[-2:]
        batch_size = img.shape[0]

        # get target tensors
        params = self.generator(batch_size, img_size,  device=img.device, dtype=img.dtype)

        # apply transform
        img = self.apply(img, params)

        return img

# --------------------------------------
#             Geometric
# --------------------------------------


class RandomHorizontalFlip(AugmentationBase):
    r"""Applies a random horizontal flip to a tensor image or a batch of tensor images with a given probability.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    """

    def __init__(self, p=0.2):
        super(RandomHorizontalFlip, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):

        target = self._adapted_sampling(batch_size, device, dtype)

        # params
        params = dict()
        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = hflip(inp)[target]
        return out


class RandomVerticalFlip(AugmentationBase):

    r"""Applies a random vertical flip to a tensor image or a batch of tensor images with a given probability.
    """
    def __init__(self, p=0.2):
        super(RandomVerticalFlip, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):
        target = self._adapted_sampling(batch_size, device, dtype)

        # params
        params = dict()
        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = vflip(inp)[target]
        return out


class RandomPerspective(AugmentationBase):
    r"""Applies a random perspective transformation to an image tensor with a given probability.

    """

    def __init__(self, p, distortion_scale, interpolation='bilinear', border_mode='zeros', align_corners=False):
        super(RandomPerspective, self).__init__(p=p)
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, device, dtype):

        target = self._adapted_sampling(batch_size, device, dtype)

        height, width = img_size

        distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)

        assert distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1, \
            f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}."

        assert type(height) == int and height > 0 and type(width) == int and width > 0, \
            f"'height' and 'width' must be integers. Got {height}, {width}."

        start_points = torch.tensor([[
            [0., 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]], device=device, dtype=dtype).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = distortion_scale * width / 2
        fy = distortion_scale * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

        pts_norm = torch.tensor([[
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]], device=device, dtype=dtype)

        rand_val = _adapted_uniform(start_points.shape,
                                    torch.tensor(0, device=device, dtype=dtype),
                                    torch.tensor(1, device=device, dtype=dtype)
                                    ).to(device=device, dtype=dtype)

        end_points = start_points + factor * rand_val * pts_norm

        params = dict()
        params["target"] = target
        params["start_points"] = start_points
        params["end_points"] = end_points

        return params

    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        transform = get_perspective_transform(params['start_points'], params['end_points']).type_as(inp)

        size = inp.shape[-2:]

        out[target] = warp_perspective(inp, transform, size,
                                       interpolation=self.interpolation,
                                       border_mode=self.border_mode,
                                       align_corners=self.align_corners)[target]

        return out


class RandomAffine(AugmentationBase):
    r"""Applies a random 2D affine transformation to a tensor image.

    The transformation is computed so that the image center is kept invariant.
    """

    def __init__(self, p, theta, h_trans, v_trans, scale, shear,
                 interpolation='bilinear', padding_mode='zeros', align_corners=False):
        super(RandomAffine, self).__init__(p=p)
        self.theta = [-theta, theta]
        self.translate = [h_trans, v_trans]
        self.scale = scale
        self.shear = shear

        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, device, dtype):

        height, width = img_size

        target = self._adapted_sampling(batch_size, device, dtype)

        assert isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0, \
            f"`width` and `height` must be positive integers. Got {width}, {height}."

        degrees = torch.as_tensor(self.theta).to(device=device, dtype=dtype)
        angle = _adapted_uniform((batch_size,), degrees[0], degrees[1]).to(device=device, dtype=dtype)

        # compute tensor ranges
        if self.scale is not None:
            scale = torch.as_tensor(self.scale).to(device=device, dtype=dtype)

            assert len(scale.shape) == 1 and (len(scale) == 2), \
                f"`scale` shall have 2 or 4 elements. Got {scale}."

            _scale = _adapted_uniform((batch_size,), scale[0], scale[1]).unsqueeze(1).repeat(1, 2)

        else:
            _scale = torch.ones((batch_size, 2), device=device, dtype=dtype)

        if self.translate is not None:
            translate = torch.as_tensor(self.translate).to(device=device, dtype=dtype)

            max_dx = translate[0] * width
            max_dy = translate[1] * height

            translations = torch.stack([
                _adapted_uniform((batch_size,), max_dx * 0, max_dx),
                _adapted_uniform((batch_size,), max_dy * 0, max_dy)
            ], dim=-1).to(device=device, dtype=dtype)

        else:
            translations = torch.zeros((batch_size, 2), device=device, dtype=dtype)

        center = torch.tensor([width, height], device=device, dtype=dtype).view(1, 2) / 2. - 0.5
        center = center.expand(batch_size, -1)

        if self.shear is not None:
            shear = torch.as_tensor(self.shear).to(device=device, dtype=dtype)

            sx = _adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)
            sy = _adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)

            sx = sx.to(device=device, dtype=dtype)
            sy = sy.to(device=device, dtype=dtype)
        else:
            sx = sy = torch.tensor([0] * batch_size, device=device, dtype=dtype)

        # params
        params = dict()
        params["target"] = target
        params["translations"] = translations
        params["center"] = center
        params["scale"] = _scale
        params["angle"] = angle
        params["sx"] = sx
        params["sy"] = sy
        return params

    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        # concatenate transforms
        transform = get_affine_matrix2d(translations=params["translations"],
                                        center=params["center"],
                                        scale=params["scale"],
                                        angle=params["angle"],
                                        sx=params["sx"],
                                        sy=params["sy"]).type_as(inp)
        size = inp.shape[-2:]

        out[target] = warp_affine(inp, transform, size,
                                  interpolation=self.interpolation,
                                  padding_mode=self.padding_mode,
                                  align_corners=self.align_corners)[target]

        return out


class RandomRotation(AugmentationBase):
    r"""Applies a random rotation to a tensor image or a batch of tensor images given an amount of degrees.
    """

    def __init__(self, p, theta, interpolation='bilinear', padding_mode='zeros', align_corners=False):
        super(RandomRotation, self).__init__(p=p)

        self.theta = [-theta, theta]

        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, device, dtype):
        r"""Get parameters for ``rotate`` for a random rotate transform.

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        angle = torch.as_tensor(self.theta).to(device=device, dtype=dtype)
        angle = _adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)

        # params
        params = dict()
        params["target"] = target
        params["angle"] = angle

        return params

    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        out[target] = rotate(inp,
                             angle=params["angle"],
                             align_corners=self.align_corners,
                             interpolation=self.interpolation)[target]

        return out


# --------------------------------------
#             Photometric
# --------------------------------------

class ColorJitter(AugmentationBase):

    def __init__(self, p, brightness=None, contrast=None, saturation=None, hue=None):
        super(ColorJitter, self).__init__(p=p)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def generator(self, batch_size, img_size, device, dtype):
        r"""Generate random color jiter parameters for a batch of images.
        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.brightness is not None:
            brightness = torch.as_tensor(self.brightness).to(device=device, dtype=dtype)
        else:
            brightness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.contrast is not None:
            contrast = torch.as_tensor(self.contrast).to(device=device, dtype=dtype)
        else:
            contrast = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.saturation is not None:
            saturation = torch.as_tensor(self.saturation).to(device=device, dtype=dtype)
        else:
            saturation = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.hue is not None:
            hue = torch.as_tensor(self.hue).to(device=device, dtype=dtype)
        else:
            hue = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        brightness_factor = _adapted_uniform((batch_size,), brightness[0], brightness[1]).to(device=device, dtype=dtype)
        contrast_factor = _adapted_uniform((batch_size,), contrast[0], contrast[1]).to(device=device, dtype=dtype)
        #saturation_factor = _adapted_uniform((batch_size,), saturation[0], saturation[1]).to(device=device, dtype=dtype)
        #hue_factor = _adapted_uniform((batch_size,), hue[0], hue[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["brightness_factor"] = brightness_factor
        params["contrast_factor"] = contrast_factor
        #params["hue_factor"] = hue_factor
        #params["saturation_factor"] = saturation_factor

        #params["order"] = torch.randperm(4, device=device, dtype=dtype).long()
        params["order"] = torch.randperm(2, device=device, dtype=dtype).long()
        params["target"] = target

        return params

    def apply(self, inp, params):

        transforms = [
            lambda img: adjust_brightness(img, brightness_factor=params["brightness_factor"]),
            lambda img: adjust_contrast(img, contrast_factor=params["contrast_factor"]),
            #lambda img: adjust_saturation(img, saturation_factor=params["saturation_factor"]),
            #lambda img: adjust_hue(img, hue_factor=params["hue_factor"])
        ]

        out = inp.clone()
        target = params["target"]

        for idx in params['order'].tolist():

            transformation = transforms[idx]
            out[target] = transformation(inp)[target]

        return out


class RandomSolarize(AugmentationBase):
    r"""Solarize given tensor image or a batch of tensor images randomly.
    """
    def __init__(self, p, thresholds=0.1, additions=0.1):
        super(RandomSolarize, self).__init__(p=p)

        self.thresholds = thresholds
        self.additions = additions

    def generator(self, batch_size, img_size, device, dtype):

        r"""Generate random solarize parameters for a batch of images.
        For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the pixel value
        to be between 0 and 1.0

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        if self.thresholds is not None:
            thresholds = torch.as_tensor(self.thresholds).to(device=device, dtype=dtype)
        else:
            thresholds = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.additions is not None:
            additions = torch.as_tensor(self.additions).to(device=device, dtype=dtype)
        else:
            additions = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        thresholds = _adapted_uniform((batch_size,), thresholds[0], thresholds[1]).to(device=device, dtype=dtype)
        additions = _adapted_uniform((batch_size,), additions[0], additions[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["thresholds"] = thresholds
        params["additions"] = additions

        params["target"] = target

        return params

    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        out[target] = solarize(inp,
                               thresholds=params["thresholds"],
                               additions=params["additions"])[target]

        return out


class RandomPosterize(AugmentationBase):
    r"""Posterize given tensor image or a batch of tensor images randomly.
    """
    def __init__(self, p, bits=3):
        super(RandomPosterize, self).__init__(p=p)
        self.bits = bits

    def generator(self, batch_size, img_size, device, dtype):
        r"""Generate random posterize parameters for a batch of images.

        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.bits is not None:
            bits = torch.as_tensor(self.bits).to(device=device, dtype=dtype)
        else:
            bits = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        bits = _adapted_uniform((batch_size,), bits[0], bits[1]).to(device=device, dtype=dtype).int()

        # Params
        params = dict()

        params["bits"] = bits
        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]

        out[target] = posterize(inp, bits=params["bits"])[target]

        return out


class RandomSharpness(AugmentationBase):
    r"""Sharpen given tensor image or a batch of tensor images randomly.

    """
    def __init__(self, p, sharpness=0.5):
        super(RandomSharpness, self).__init__(p=p)
        self.sharpness = sharpness

    def generator(self, batch_size, img_size, device, dtype):
        r"""Generate random sharpness parameters for a batch of images.

        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.sharpness is not None:
            sharpness = torch.as_tensor(self.sharpness).to(device=device, dtype=dtype)
        else:
            sharpness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        sharpness = _adapted_uniform((batch_size,), sharpness[0], sharpness[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["sharpness"] = sharpness
        params["target"] = target

        return params

    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        out[target] = sharpness(inp, sharpness_factor=params["sharpness"])[target]
        return out


class RandomEqualize(AugmentationBase):
    def __init__(self, p):
        super(RandomEqualize, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):
        r"""Generate random Equalize parameters for a batch of images.

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        # Params
        params = dict()

        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]

        out[target] = equalize(inp)[target]
        return out


class RandomGrayscale(AugmentationBase):
    r"""Applies random transformation to Grayscale according to a probability p value.
    """
    def __init__(self, p=0.1):
        super(RandomGrayscale, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):
        r"""Generate random Equalize parameters for a batch of images.

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        # Params
        params = dict()

        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]

        out[target] = grayscale(inp)[target]
        return out

# --------------------------------------
#             Filters
# --------------------------------------


class RandomMotionBlur(AugmentationBase):
    r"""Perform motion blur on 2D images (4D tensor).
    """

    def __init__(self, p, kernel_size, angle, direction,
            interpolation='bilinear', border_mode='zeros', align_corners=False) -> None:
        super(RandomMotionBlur, self).__init__(p=p)

        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, device, dtype):

        r"""Get parameters for motion blur.

        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.kernel_size is not None:
            kernel_size = torch.as_tensor(self.kernel_size).to(device=device, dtype=dtype)
        else:
            kernel_size = torch.as_tensor([3, 3]).to(device=device, dtype=dtype)

        if self.angle is not None:
            angle = torch.as_tensor(self.angle).to(device=device, dtype=dtype)
        else:
            angle = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

        if self.direction is not None:
            direction = torch.as_tensor(self.direction).to(device=device, dtype=dtype)
        else:
            direction = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

        kernel_size = _adapted_uniform((1,),
                                       torch.div(kernel_size[0], 2, rounding_mode='floor'),
                                       torch.div(kernel_size[1], 2, rounding_mode='floor')).to(device=device, dtype=dtype).int() * 2 + 1

        angle = _adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)
        direction = _adapted_uniform((batch_size,), direction[0], direction[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["kernel_size"] = kernel_size
        params["angle"] = angle
        params["direction"] = direction

        params["target"] = target

        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = motion_blur(inp,
                                  kernel_size=params["kernel_size"],
                                  angle=params["angle"],
                                  direction=params["direction"])[target]
        return out


class RandomGaussianBlur(AugmentationBase):
    r"""Perform motion blur on 2D images (4D tensor).
    """

    def __init__(self, p, kernel_size, sigma,
            interpolation='bilinear', border_mode='zeros', align_corners=False) -> None:
        super(RandomGaussianBlur, self).__init__(p=p)

        self.kernel_size = kernel_size
        self.sigma = sigma

        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, device, dtype):

        r"""Get parameters for motion blur.

        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.kernel_size is not None:
            kernel_size = torch.as_tensor(self.kernel_size).to(device=device, dtype=dtype)
        else:
            kernel_size = torch.as_tensor([3, 3]).to(device=device, dtype=dtype)

        if self.sigma is not None:
            sigma = torch.as_tensor(self.sigma).to(device=device, dtype=dtype)
        else:
            sigma = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

        kernel_size = _adapted_uniform((1,),
                                       torch.div(kernel_size[0], 2, rounding_mode='floor'),
                                       torch.div(kernel_size[1], 2, rounding_mode='floor')).to(device=device, dtype=dtype).int() * 2 + 1

        sigma = _adapted_uniform((batch_size,), sigma[0], sigma[1]).to(device=device, dtype=dtype)


        # Params
        params = dict()

        params["kernel_size"] = (int(kernel_size), int(kernel_size))
        params["sigma"] = (sigma, sigma)
        params["target"] = target

        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = gaussian_blur2d(inp,
                                  kernel_size=params["kernel_size"],
                                  sigma=params["sigma"])[target]
        return out
