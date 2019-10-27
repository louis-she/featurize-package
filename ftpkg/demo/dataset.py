from featurize_jupyterlab.core import Dataset, Option
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """This is a simple wrap for torchvision.datasets.MNIST
    """
    fold = Option(help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")

    def __call__(self):
        return (
            datasets.MNIST(self.fold, download=True, train=True, transform=transforms.ToTensor()),
            datasets.MNIST(self.fold, download=True, train=False, transform=transforms.ToTensor())
        )
