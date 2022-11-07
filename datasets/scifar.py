import PIL.Image

import torchvision

import torch.utils.data

from torchvision.datasets import CIFAR10


class SequentialCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.data = CIFAR10(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)
        return image.reshape(-1), label
