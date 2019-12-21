from featurize_jupyterlab.core import Loss, Option
from minetorch.loss import *
import torch.nn.functional as F
import torch

class BCEWithLogitsLoss(Loss):
    """PyTorch's BCEWithLogitsLoss
    """
    
    def __call__(self):
        return torch.nn.BCEWithLogitsLoss()

class DiceLoss(Loss):
    """Dice Loss
    """

    def __call__(self):
        return dice_loss