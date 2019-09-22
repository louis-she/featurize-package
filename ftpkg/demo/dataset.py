from featurize_jupyterlab.core import dataset, option
from torchvision import datasets


@dataset('MNIST Train', 'This is a simple wrap for torchvision.datasets.MNIST(train=True)')
@option('fold', help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")
def mnist_train(fold):
    return datasets.MNIST(fold, download=True, train=True)

@dataset('MNIST Test', 'This is a simple wrap for torchvision.datasets.MNIST(Train=False)')
@option('fold', help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")
def mnist(fold):
    return datasets.MNIST(fold, download=True, train=False)
