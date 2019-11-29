from featurize_jupyterlab.core import Model, Option
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
import segmentation_models_pytorch as smp

encoder_weights_collection = [('random', None), 'imagenet']
activation_collection = ['None', 'sigmoid', 'softmax']
architecture_collection = ['Unet', 'FPN', 'Linknet', 'PSPNet']
encoder_name_collection = [
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'inceptionresnetv2',
    'inceptionv4',
    'mobilenet_v2',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
    'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    'xception'
    ]



class SegmentationModel(Model):

    encoder_name = Option(default='resnet34', type='collection', collection=encoder_name_collection)
    encoder_weights = Option(default='imagenet', type='collection', collection=encoder_weights_collection)
    num_classes = Option(default=2, type='number', help='class number of mask')
    activation = Option(default='None', type='collection', collection=activation_collection)
    model_architecture = Option(defalt='Unet', type='collection', collection=architecture_collection)
    
    def create_model(self):
        kwargs = {
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights, 
            'classes':self.num_classes, 
            'activation':self.activation
        }
        if model_architecture == 'Unet':
            model = smp.Unet(**kwargs)
        elif model_architecture == 'FPN':
            model = smp.FPN(**kwargs)
        elif model_architecture == 'Linknet':
            model = smp.Linknet(**kwargs)
        elif model_architecture == 'PSPNet':
            model = smp.Linknet(**kwargs)
        
        model.eval()
        
        return model
