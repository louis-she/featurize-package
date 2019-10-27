from featurize_jupyterlab.core import Optimizer, Option
from torch.optim import SGD


class PyTorchSGD(Optimizer):

    lr = Option(default=0.1)

    def __call__(self):
        return SGD(self.model.parameters(), float(self.lr))
