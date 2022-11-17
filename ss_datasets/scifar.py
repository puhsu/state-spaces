from typing import Callable

import PIL.Image

import torchvision

import torch.utils.data

from torchvision.datasets import CIFAR10

from skimage.color import rgb2gray


class SequentialCIFAR10(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
            download: bool = False,
            grayscale: bool = False,
            tokenize: bool = False
    ):
        self.data = CIFAR10(root, train, transform, target_transform, download)
        self._grayscale = grayscale
        self._tokenize = tokenize

        print(f'Loading sCIFAR with tokenize={tokenize} and grayscale={grayscale}')

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if isinstance(image, PIL.Image.Image):
            if self._grayscale:
                image = rgb2gray(image)
                image = torchvision.transforms.ToTensor()(image)
                image = image.squeeze(0)  # remove redundant channel dim
                image = image.reshape(-1)
            else:
                image = torchvision.transforms.ToTensor()(image)
                image = image.transpose(0, 2).reshape(-1, image.shape[0])

        if self._tokenize:
            image = (image * 255).to(torch.uint8)

        # Convert to shape [WIDTH * HEIGHT, N_CHANNELS]
        return image, label

    def __len__(self):
        return len(self.data)
