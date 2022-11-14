from typing import Callable

import PIL.Image

import torchvision

import torch.utils.data

from torchvision.datasets import CIFAR10


class SequentialCIFAR10(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
            download: bool = False
    ):
        self.data = CIFAR10(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        # Convert to shape [WIDTH * HEIGHT, N_CHANNELS]
        return image.transpose(0, 2).reshape(-1, image.shape[0]), label

    def __len__(self):
        return len(self.data)