from typing import Callable

import PIL.Image

import torchvision

import torch.utils.data

from torchvision.datasets import MNIST


class SequentialMNIST(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
            download: bool = False,
            task=None
    ):
        self.task = task
        self.data = MNIST(root, train, transform, target_transform, download)

        self.in_features = 1
        self.d_output = (1, 256) if self.task == 'density_estimation' else 10

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        # Convert to shape [WIDTH * HEIGHT, N_CHANNELS]
        image, label = image.transpose(0, 2).reshape(-1, image.shape[0]), label
        if self.task == 'density_estimation':
            return image[:-1], (255 * image[1:]).to(torch.long)
        else:
            return image, label

    def __len__(self):
        return len(self.data)
