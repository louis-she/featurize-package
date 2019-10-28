from featurize_jupyterlab.core import Dataset, Option
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """This is a simple wrap for torchvision.datasets.MNIST
    """
    fold = Option(help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")

    def __call__(self):
        return (
            DataLoader(dataset=datasets.MNIST(self.fold, download=True, train=True, transform=transforms.ToTensor()), batch_size=128),
            DataLoader(dataset=datasets.MNIST(self.fold, download=True, train=False, transform=transforms.ToTensor()), batch_size=128)
        )
