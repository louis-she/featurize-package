from featurize_jupyterlab.transform import DualImageTransformation
from albumentations import HorizontalFlip


class FeaturizeHorizontalFlip(DualImageTransformation):
    """Apply random crop to images, masks, keypoints and bounding boxes
    """
    def create_aug(self):
        return HorizontalFlip(
            p=self.probability  # again, probability is predefined
        )
