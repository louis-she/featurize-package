from featurize_jupyterlab.core import Loss
import torch.nn.functional as F


class nll_loss(Loss):
    """Simple wrap of PyTorch's nll_loss
    """

    def __call__(self, data):
        image, target = data
        output = self.trainer.model(image)
        return F.nll_loss(output, target)
