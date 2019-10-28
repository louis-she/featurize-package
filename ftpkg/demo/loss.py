from featurize_jupyterlab.core import Loss
import torch.nn.functional as F


class NllLoss(Loss):
    """Simple wrap of PyTorch's nll_loss
    """

    def __call__(self):
        return F.nll_loss
