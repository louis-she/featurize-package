import json

from albumentations import HorizontalFlip

from featurize_jupyterlab.core import Option
from featurize_jupyterlab.transform import DualImageTransformation


class FeaturizeHorizontalFlip(DualImageTransformation):
    """Apply random crop to images, masks, keypoints and bounding boxes
    """
    columns_config = Option(type='string', default='{"image": 0, "mask": 1}', post_process=lambda x: json.loads(x))

    def create_aug(self):
        return HorizontalFlip(
            p=self.probability  # again, probability is predefined
        )
