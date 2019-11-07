from featurize_jupyterlab.core import Model, Option
from .unet import Unet
from efficientnet_pytorch.model import EfficientNet

encoder_weights_collection = [('random', None), 'imagenet']
encoder_name_collection = [
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
    'efficientnet-b4',
    'efficientnet-b5',
    'efficientnet-b6',
    'efficientnet-b7'
    ]


class Efficientnet(Model):
    """Unet is a fully convolution neural network for image semantic segmentation
    """
    backbone = Option(default='efficientnet-b0', type='collection', collection=encoder_name_collection)
    class_number = Option(default=6, type='number', help='class number')

    def __call__(self):
        return EfficientNet.from_pretrained(
            self.backbone,
            num_classes=self.class_number
            )
