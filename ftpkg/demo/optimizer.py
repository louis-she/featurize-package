from feautrize_jupyterlab.core import optimizer, option, g
from torch.optim import SGD


@optimizer('PyTorch SGD', 'Simple wrap of the SGD optimizer of PyTorch')
@option('lr', help='Learning Rate', type='number', required=True, default='0.1')
def sgd(lr):
    return SGD(g.model.parameters(), float(lr))
