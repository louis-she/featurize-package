from featurize_jupyterlab.core import Model
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
import segmentation_models_pytorch as smp

encoder_weights_collection = [('random', None), 'imagenet']
encoder_name_collection = ['resnet34', 'resnet50', 'resnet101']
activation_collection = ['None', 'sigmoid', 'softmax']

encoder_name = Option(default='resnet34', type='collection', collection=encoder_name_collection)
encoder_weights = Option(default='imagenet', type='collection', collection=encoder_weights_collection)
num_classes = Option(default=2, type='number', help='class number of mask')
activation = Option(default='None', type='collection', collection=activation_collection)

class FPNResnet34(Model):

    def __call__(self):
        model = smp.FPN(
            encoder_name=self.encoder_name, 
            encoder_weights=self.encoder_weights, 
            classes=self.num_classes, 
            activation=self.activation,
        )
        model.eval()
        return model
