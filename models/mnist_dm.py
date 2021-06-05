import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

"""
Sample DataModule for MNIST Dataset.
"""


class MNISTData(pl.LightningDataModule):

    def __init__(self, data_dir='./data', batch_size=128,
                 num_workers=2, shuffle_train=True):

        super(MNISTData, self).__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.nb_workers = num_workers
        self.shuffle = shuffle_train

        self.train_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        mnist_train = MNIST(self.data_dir, train=True,
                            download=True, transform=self.train_transforms)

        self.train, self.val = random_split(mnist_train,
                                            [56000, 4000])
        self.test = MNIST(self.data_dir, train=False,
                          download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.nb_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.nb_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.nb_workers)
