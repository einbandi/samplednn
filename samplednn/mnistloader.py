from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import os

def setup_mnist_loader(batch_size=128, use_cuda=False, train=True,
    shuffle=False, transform=ToTensor()):

    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..',
        'data'
    ))

    dset = MNIST(
        data_dir,
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=use_cuda
    )
