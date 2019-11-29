# DualImageTransformation
# Spatial-level transforms for both images, masks, bbox and keypoints
import albumentations.augmentations.transforms as albu
import json
import cv2
import torchvision.transforms as transform

from featurize_jupyterlab.core import Option
from featurize_jupyterlab.transform import BasicImageTransformation, DualImageTransformation

class CenterCrop(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='resize width of target image')
    height = Option(type='number', help='resize height of target image')

    def create_aug(self):
        return albu.CenterCrop(self.height, self.width, p=self.probability)


class Crop(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    x_min = Option(type='number', help='minimum upper left x coordinate.')
    y_min = Option(type='number', help='minimum upper left y coordinate.')
    x_max = Option(type='number', help='maximum lower right x coordinate')
    y_max = Option(type='number', help='maximum lower right y coordinate')

    def create_aug(self):
        return albu.CenterCrop(self.x_min, self.y_min, self.x_max, self.y_max, p=self.probability)


class CropNonEmptyMaskIfExists(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='resize width of target image')
    height = Option(type='number', help='resize height of target image')
    ignore_values = Option(type='boolean', help='ignore backgroud in mask')
    ignore_channels = Option(type='number', help='channels to ignore in mask, eg.if background is a first channel set "ignore_channels=1" to ignore')

    def create_aug(self):
        if self.ignore_values == True:
            self.ignore_values = [0]
        else:
            self.ignore_values = None

        if isinstance(a,int):
            self.ignore_channels = self.ignore_channels - 1
        else:
            self.ignore_channels = None

        return albu.CenterCrop(
            self.height,
            self.width,
            ignore_values=self.ignore_values,
            ignore_channels=self.ignore_channels,
            p=self.probability)


class Flip(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.Flip(self.probability)


class HorizontalFlip(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.HorizontalFlip(self.probability)


class IAAAffine(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAAffine(
            scale=1.0,
            translate_percent=None,
            translate_px=None,
            rotate=0.0, shear=0.0,
            order=1,
            cval=0,
            mode='reflect',
            always_apply=False,
            p=self.probability)


class IAACropAndPad(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAACropAndPad(
            px=None,
            percent=None,
            pad_mode='constant',
            pad_cval=0,
            keep_size=True,
            always_apply=False,
            p=self.probability)


class IAAFliplr(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAFliplr(self.probability)


class IAAFlipud(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAFlipud(self.probability)


class IAAPerspective(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAPerspective(
            scale=(0.05, 0.1),
            keep_size=True,
            always_apply=False,
            p=self.probability)


class IAAPiecewiseAffine(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAPiecewiseAffine(
            scale=(0.03, 0.05),
            nb_rows=4,
            nb_cols=4,
            order=1,
            cval=0,
            mode='constant',
            always_apply=False,
            p=self.probability)


class PadIfNeeded(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    min_height = Option(type='number', help='minimal result image height')
    min_width = Option(type='number', help='minimal result image width')

    def create_aug(self):
        return albu.PadIfNeeded(
            min_height=self.min_height,
            min_width=self.min_width,
            border_mode=4,
            value=None,
            mask_value=None,
            always_apply=False,
            p=self.probability
            )


class RandomCrop(DualImageTransformation):
    """Randomly crop an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='width to crop')
    height = Option(type='number', help='height to crop')

    def create_aug(self):
        return albu.RandomCrop(self.height, self.width, p=self.probability)


class RandomCropNearBBox(DualImageTransformation):
    """Randomly crop an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    max_part_shift = Option(type='number', help='float value in (0.0, 1.0) range')

    def create_aug(self):
        return albu.RandomCropNearBBox(
            max_part_shift=self.max_part_shift,
            always_apply=False,
            p=self.probability
            )


class RandomResizedCrop(DualImageTransformation):
    """Randomly crop an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='resize width of target image')
    height = Option(type='number', help='resize height of target image')

    def create_aug(self):
        return albu.RandomResizedCrop(
            height=self.height,
            weidth=self.width,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=1,
            always_apply=False,
            p=self.probability
            )


class RandomRotate90(DualImageTransformation):
    """Randomly crop an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.RandomRotate90(always_apply=False, p=self.probability)


class RandomSizedCrop(DualImageTransformation):
    """Randomly crop an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    min_height = Option(type='number', help='minimum height limit of image')
    max_height = Option(type='number', help='maximum height limit of image')
    height = Option(type='number', help='height after crop and resize.')
    width = Option(type='number', help='width after crop and resize.')

    def create_aug(self):
        return albu.RandomSizedCrop(
            min_max_height=(self.min_height, self.max_height),
            height=self.height,
            width=self.width,
            w2h_ratio=1.0,
            interpolation=1,
            always_apply=False,
            p=self.probability
            )


class Resize(DualImageTransformation):
    """
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='resize width of target image')
    height = Option(type='number', help='resize height of target image')

    def create_aug(self):
        return albu.Resize(self.height, self.width)


class RandomHorizontalFlip(DualImageTransformation):
    """Randomly horizontal flip an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.HorizontalFlip(p=self.probability)


class VerticalFlip(DualImageTransformation):
    """Randomly horizontal flip an image
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.VerticalFlip(always_apply=False, p=self.probability)


# BasicImageTransformation
# Pixel-level transforms for images only

class Blur(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    blur_limit = Option(type='number', default=7, help='maximum kernel size for blurring the input image.')

    def create_aug(self):
        return albu.Resize(
            blur_limit=self.blur_limit,
            p=self.probability
            )


class Blur(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    clip_limit = Option(type='number', default=4.0, help='upper threshold value for contrast limiting.')
    def create_aug(self):
        return albu.CLAHE(
            clip_limit=self.clip_limit,
            tile_grid_size=(8, 8),
            always_apply=False,
            p=self.probability
            )


class ChannelDropout(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    channel_drop = Option(type='number', default=1, help='upper limit for channels to be drop')
    def create_aug(self):
        return albu.ChannelDropout(
            channel_drop_range=(1, self.channel_drop),
            fill_value=0,
            always_apply=False,
            p=self.probability
            )


class ChannelShuffle(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.ChannelShuffle(always_apply=False, p=self.probability)


class CoarseDropout(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    max_holes = Option(type='number', default=8, help='Maximum number of regions to zero out.')
    max_height = Option(type='number', default=8, help='Maximum height of the hole.')
    max_width = Option(type='number', default=8, help='Maximum width of the hole.')
    def create_aug(self):
        return albu.CoarseDropout(
            max_holes=self.max_holes,
            max_height=self.max_height,
            max_width=self.max_width,
            min_holes=None,
            min_height=None,
            min_width=None,
            fill_value=0,
            always_apply=False,
            p=self.probability
            )


class Cutout(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    num_holes = Option(type='number', default=8, help='Number of regions to zero out.')
    max_h_size = Option(type='number', default=8, help='Maximum height of the hole.')
    max_w_size = Option(type='number', default=8, help='Maximum width of the hole.')
    def create_aug(self):
        return albu.Cutout(
            num_holes=self.num_holes,
            max_h_size=self.max_h_size,
            max_w_size=self.max_w_size,
            fill_value=0,
            always_apply=False,
            p=self.probability
            )


class Downscale(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    scale_min = Option(type='number', default=0.25, help='Lower bound on the image scale. Should be < 1.')
    scale_max = Option(type='number', default=0.25, help='')
    def create_aug(self):
        return albu.Downscale(
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            interpolation=0,
            always_apply=False,
            p=0.5
            )


class Equalize(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.Equalize(
            mode='cv',
            by_channels=True,
            mask=None,
            mask_params=(),
            always_apply=False,
            p=self.probability
            )


class GaussNoise(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    lower_limit = Option(type='number', default=10.0, help='Variance lower limit for noise.')
    upper_limit = Option(type='number', default=50.0, help='Variance upper limit for noise.')
    mean = Option(type='number', default=0, help='Mean of the noise.')
    def create_aug(self):
        return albu.GaussNoise(
            var_limit=(self.lower_limit, self.upper_limit),
            mean=self.mean,
            always_apply=False,
            p=self.probability
            )


class GaussianBlur(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    blur_limit = Option(type='number', default=7, help='Maximum Gaussian kernel size for blurring the input image.')
    def create_aug(self):
        return albu.GaussianBlur(
            blur_limit=self.blur_limit,
            always_apply=False,
            p=self.probability
            )


class HueSaturationValue(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    hue_shift_limit = Option(type='number', default=20, help='Range for changing hue.(-hue_shift_limit, hue_shift_limit)')
    sat_shift_limit = Option(type='number', default=30, help='Range for changing saturation.(-sat_shift_limit, sat_shift_limit)')
    val_shift_limit = Option(type='number', default=20, help='Range for changing value.(-val_shift_limit, val_shift_limit)')
    def create_aug(self):
        return albu.HueSaturationValue(
            hue_shift_limit=self.hue_shift_limit,
            sat_shift_limit=self.sat_shift_limit,
            val_shift_limit=self.val_shift_limit,
            always_apply=False,
            p=self.probability
            )


class IAAAdditiveGaussianNoise(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    loc = Option(type='number', default=0, help='Mean of the normal distribution that generates the noise.')
    upper_limit = Option(type='number', default=12.75, help='Upper limit of standard deviation of the normal distribution that generates the noise. ')
    per_channel = Option(type='boolean', default=False)
    def create_aug(self):
        return albu.IAAAdditiveGaussianNoise(
            loc=self.loc,
            scale=(0., self.upper_limit),
            per_channel=self.per_channel,
            always_apply=False,
            p=self.probability
            )


class IAAEmboss(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    alpha = Option(type='number', default=(0.2, 0.5), help='Range to choose the visibility of the embossed image. At 0, only the original image is visible,at 1.0 only its embossed version is visible.')
    strength = Option(type='number', default=(0.2, 0.7), help='Strength range of the embossing.')
    def create_aug(self):
        return albu.IAAEmboss(
            alpha=(0.2, 0.5),
            strength=(0.2, 0.7),
            always_apply=False,
            p=self.probability
            )


class IAASharpen(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    alpha = Option(type='number', default=(0.2, 0.5), help='Range to choose the visibility of the embossed image. At 0, only the original image is visible,at 1.0 only its embossed version is visible.')
    lightness = Option(type='number', default=(0.5, 1.0), help='Range to choose the lightness of the sharpened image.')
    def create_aug(self):
        return albu.IAASharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            always_apply=False,
            p=self.probability
            )


class IAASuperpixels(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    p_replace = Option(type='number', default=0.1, help='Defines the probability of any superpixel area being replaced by the superpixel, i.e. by the average pixel color within its area. ')
    n_segments = Option(type='number', default=100, help='Target number of superpixels to generate. ')
    def create_aug(self):
        return albu.IAASuperpixels(
            p_replace=self.p_replace,
            n_segments=self.n_segments,
            always_apply=False,
            p=self.probability
            )


class ISONoise(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            always_apply=False,
            p=self.probability
            )


class ImageCompression(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    quality_lower = Option(type='number', default=99, help='Lower bound on the image quality. ')
    quality_upper  = Option(type='number', default=100, help='Upper bound on the image quality. ')
    def create_aug(self):
        return albu.ImageCompression(
            quality_lower=self.quality_lower,
            quality_upper=self.quality_upper,
            always_apply=False,
            p=self.probability
            )


class InvertImg(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.InvertImg(
            always_apply=False,
            p=self.probability
            )


class JpegCompression(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    quality_lower  = Option(type='number', default=99, help='Lower bound on the jpeg quality.')
    quality_upper  = Option(type='number', default=100, help='Upper bound on the jpeg quality. ')
    def create_aug(self):
        return albu.JpegCompression(
            quality_lower=self.quality_lower,
            quality_upper=self.quality_upper,
            always_apply=False,
            p=self.probability
            )


class MedianBlur(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    blur_limit  = Option(type='number', default=7, help='Maximum aperture linear size for blurring the input image. ')
    def create_aug(self):
        return albu.MedianBlur(
            blur_limit=7,
            always_apply=False,
            p=self.probability
            )


class MotionBlur(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    blur_limit  = Option(type='number', default=7, help='Maximum kernel size for blurring the input image. ')
    def create_aug(self):
        return albu.MotionBlur(
            blur_limit=self.blur_limit,
            always_apply=False,
            p=self.probability
            )


class Normalize(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    normalize_type = Option(type='collection', default='imagenet', collection=[['imagenet', 'imagenet']])

    def create_aug(self):
        if self.normalize_type == 'imagenet':
            params = {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225)
            }
        else:
            params = {

            }
        return albu.Normalize(**params, p=self.probability)


class Posterize(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.Posterize(
            num_bits=4,
            always_apply=False,
            p=self.probability
            )


class RGBShift(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    r_shift_limit  = Option(type='number', default=20, help='Range for changing values for the red channel. If r_shift_limit is a single int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).')
    g_shift_limit  = Option(type='number', default=20, help='Range for changing values for the green channel. If g_shift_limit is a single int, the range will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).')
    b_shift_limit  = Option(type='number', default=20, help='Range for changing values for the blue channel. If b_shift_limit is a single int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).')
    def create_aug(self):
        return albu.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            always_apply=False,
            p=self.probability
            )


class RandomBrightness(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    limit = Option(type='number', default=0.2, help='Factor range for changing brightness. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).')
    def create_aug(self):
        return albu.RandomBrightness(
            limit=self.limit,
            always_apply=False,
            p=self.probability
            )


class RandomContrast(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    limit = Option(type='number', default=0.2, help='Factor range for changing contrast. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).')
    def create_aug(self):
        return albu.RandomContrast(
            limit=self.limit,
            always_apply=False,
            p=self.probability
            )


class RandomFog(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    fog_coef_lower = Option(type='number', default=0.3, help='Lower limit for fog intensity coefficient. ')
    fog_coef_upper = Option(type='number', default=1, help='Upper limit for fog intensity coefficient. ')
    alpha_coef = Option(type='number', default=0.08, help='Transparency of the fog circles. ')
    def create_aug(self):
        return albu.RandomFog(
            fog_coef_lower=0.3,
            fog_coef_upper=1,
            alpha_coef=0.08,
            always_apply=False,
            p=self.probability
            )


class RandomGamma(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    def create_aug(self):
        return albu.RandomGamma(
            gamma_limit=(80, 120),
            eps=1e-07,
            always_apply=False,
            p=self.probability
            )


class ToGray(BasicImageTransformation):
    """
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))

    def create_aug(self):
        def _transform(image):
            return {'image': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)}
        return _transform


class ToTensor(BasicImageTransformation):
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))

    def create_aug(self):
        to_tensor = transform.ToTensor()

        def _transform(image):
            return {'image': to_tensor(image)}
        return _transform
