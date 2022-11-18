"""Long Range Arena ss_datasets"""
import io
import logging
import os
import pickle
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchtext
import torchvision
from einops.layers.torch import Rearrange, Reduce
from PIL import Image  # Only used for Pathfinder
from datasets import DatasetDict, Value, load_dataset

from ss_datasets.base import default_data_path, SequenceDataset, ImageResolutionSequenceDataset


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class PathFinder(ImageResolutionSequenceDataset):
    _name_ = "pathfinder"
    d_input = 1
    d_output = 2
    l_output = 0

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 32,
            "sequential": True,
            "tokenize": False,
            "pool": 1,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
        }

    def default_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize
                else Rearrange("1 h w -> (h w) 1")
            )
        else:
            transform_list.append(Rearrange("1 h w -> h w 1"))
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is either 32, 64, 128, or 256.
            """
            )

    def setup(self, stage=None):
        if self.data_dir is None:
            self.data_dir = (
                default_data_path / self._name_ / f"pathfinder{self.resolution}"
            )

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )
