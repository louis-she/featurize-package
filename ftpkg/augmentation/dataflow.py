from featurize_jupyterlab.core import Dataflow, Option
from featurize_jupyterlab.transform import BasicImageTransformation, DualImageTransformation
import json
import albumentations as albu

# DualImageTransformation 
# Spatial-level transforms for both images, masks, bbox and keypoints

class CenterCrop(DualImageTransformation):
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    width = Option(type='number', help='resize width of target image')
    height = Option(type='number', help='resize height of target image')

    def create_aug(self):
        return albu.CenterCrop(self.height, self.width, p=self.probability)


class Crop(DualImageTransformation):
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    x_min = Option(type='number', help='minimum upper left x coordinate.')
    y_min = Option(type='number', help='minimum upper left y coordinate.')
    x_max = Option(type='number', help='maximum lower right x coordinate')
    y_max = Option(type='number', help='maximum lower right y coordinate')

    def create_aug(self):
        return albu.CenterCrop(self.x_min, self.y_min, self.x_max, self.y_max, p=self.probability)


class CropNonEmptyMaskIfExists(DualImageTransformation):
    """Resize image to any size
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
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.Flip(self.probability)


class HorizontalFlip(DualImageTransformation):
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.HorizontalFlip(self.probability)


class IAAAffine(DualImageTransformation):
    """Resize image to any size
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
    """Resize image to any size
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
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAFliplr(self.probability)


class IAAFlipud(DualImageTransformation):
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return albu.IAAFlipud(self.probability)


class IAAPerspective(DualImageTransformation):
    """Resize image to any size
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))
    
    def create_aug(self):
        return albu.IAAPerspective(
            scale=(0.05, 0.1), 
            keep_size=True, 
            always_apply=False, 
            p=self.probability)


class IAAPiecewiseAffine(DualImageTransformation):
    """Resize image to any size
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
    """Resize image to any size
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
            p=self.probability)


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
        return albu.RandomCropNearBBox(max_part_shift=self.max_part_shift, always_apply=False, p=self.probability)


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
            p=self.probability)


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
            p=self.probability)


class Resize(DualImageTransformation):
    """Resize image to any size
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

class Normalize(BasicImageTransformation):
    """Normalize image
    """
    columns_config = Option(type='string', default='[0]', post_process=lambda x: json.loads(x))
    normalize_type = Option(type='collection', default='imagenet', collection=[['imagenet', 'imagenet']])

    def create_aug(self):
        if self.normalize_type == 'imagenet':
            params = {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225)
            }
        return albu.Normalize(**params, p=self.probability)
