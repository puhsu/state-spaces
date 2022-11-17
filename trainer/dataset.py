from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from torch.utils.data import DataLoader, random_split

from ss_datasets import SequentialCIFAR10
from ss_datasets.lra.configure import configure_lra
from ss_datasets.lra.loader import PathFinder


TOKENIZE_DATASETS = False


def set_tokenize(value: bool) -> None:
    global TOKENIZE_DATASETS
    TOKENIZE_DATASETS = value


class Dataset(Enum):
    CIFAR = 'cifar'
    PATH32 = 'path32'
    PATH64 = 'path64'
    PATH128 = 'path128'
    PATH256 = 'path256'


class PathXHandler:

    path32: Optional[PathFinder] = None
    path64: Optional[PathFinder] = None
    path128: Optional[PathFinder] = None
    path256: Optional[PathFinder] = None

    __init__ = None

    @classmethod
    def get_path32(cls) -> PathFinder:
        if cls.path32 is None:
            cls.path32 = configure_lra(x=32, tokenize=TOKENIZE_DATASETS)
        return cls.path32

    @classmethod
    def get_path64(cls) -> PathFinder:
        if cls.path64 is None:
            cls.path64 = configure_lra(x=64, tokenize=TOKENIZE_DATASETS)
        return cls.path64

    @classmethod
    def get_path128(cls) -> PathFinder:
        if cls.path128 is None:
            cls.path128 = configure_lra(x=128, tokenize=TOKENIZE_DATASETS)
        return cls.path128

    @classmethod
    def get_path256(cls) -> PathFinder:
        if cls.path256 is None:
            cls.path256 = configure_lra(x=256, tokenize=TOKENIZE_DATASETS)
        return cls.path256


TRAIN_READERS = {
    Dataset.CIFAR: lambda: SequentialCIFAR10(root='cifar', train=True, download=True, grayscale=True, tokenize=TOKENIZE_DATASETS),
    Dataset.PATH32: lambda: PathXHandler.get_path32().dataset_train,
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_train,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_train,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_train
}
VAL_READERS = {
    Dataset.PATH32: lambda: PathXHandler.get_path32().dataset_val,
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_val,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_val,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_val
}
TEST_READERS = {
    Dataset.CIFAR: lambda: SequentialCIFAR10(root='cifar', train=False, download=True, grayscale=True, tokenize=TOKENIZE_DATASETS),
    Dataset.PATH32: lambda: PathXHandler.get_path32().dataset_test,
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_test,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_test,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_test
}
NUM_FEATURES = {
    Dataset.CIFAR: 3,
    Dataset.PATH32: 1,
    Dataset.PATH64: 1,
    Dataset.PATH128: 1,
    Dataset.PATH256: 1
}
NUM_CATEGORIES = {
    Dataset.CIFAR: 10,
    Dataset.PATH32: 2,
    Dataset.PATH64: 2,
    Dataset.PATH128: 2,
    Dataset.PATH256: 2
}
VOCAB_SIZE = {
    Dataset.CIFAR: 256,
    Dataset.PATH32: 256,
    Dataset.PATH64: 256,
    Dataset.PATH128: 256,
    Dataset.PATH256: 256
}
MAX_LENGTH = {
    Dataset.CIFAR: 32 * 32,
    Dataset.PATH32: 32 * 32,
    Dataset.PATH64: 64 * 64,
    Dataset.PATH128: 128 * 128,
    Dataset.PATH256: 256 * 256
}


@dataclass
class DatasetArguments:
    dataset: Dataset = field(default=Dataset.CIFAR, metadata={'help': 'Dataset to train on.'})
    split: float = field(default=0.8, metadata={'help': 'Train/val split of official train dataset if no val dataset is available.'})
    batch_size: int = field(default=256, metadata={'help': 'Train batch size (eval batch size is doubled).'})


def get_dataloaders(args: DatasetArguments) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_reader = TRAIN_READERS[args.dataset]()
    test_reader = TEST_READERS[args.dataset]()

    if args.dataset not in VAL_READERS:
        train_length = int(args.split * len(train_reader))
        val_length = len(train_reader) - train_length
        train_reader, val_reader = random_split(train_reader, lengths=(train_length, val_length))
    else:
        val_reader = VAL_READERS[args.dataset]()

    train_batch_size = args.batch_size
    eval_batch_size = train_batch_size * 2

    train_dataloder = DataLoader(train_reader, shuffle=True, batch_size=train_batch_size)
    val_dataloder = DataLoader(val_reader, shuffle=False, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_reader, shuffle=False, batch_size=eval_batch_size)

    return train_dataloder, val_dataloder, test_dataloader
