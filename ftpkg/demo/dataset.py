from featurize_jupyterlab.core import Dataset, Option
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import datasets


class DelegateMNIST(TorchDataset):

    def __init__(self, **kwargs):
        self.transform = kwargs.pop('transform')
        self.mnist = datasets.MNIST(**kwargs)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        rowdata = list(self.mnist[index])
        return self.transform(rowdata)


class MNISTDataset(Dataset):
    """This is a simple wrap for torchvision.datasets.MNIST
    """
    fold = Option(help='Absolute folder path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")

    def __call__(self):
        return (
            DataLoader(dataset=DelegateMNIST(root=self.fold, download=True, train=True, transform=self.train_transform), batch_size=128),
            DataLoader(dataset=DelegateMNIST(root=self.fold, download=True, train=False, transform=self.val_transform), batch_size=128)
        )
