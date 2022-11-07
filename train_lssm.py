from dataclasses import field
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Dict, List, Callable

import numpy as np
import torch
from argparse_dataclass import dataclass
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import SequentialCIFAR10
from model.lssl import StateSpace
from model.s4_model import S4Model


class Dataset(str, Enum):
    CIFAR = 'cifar'


TRAIN_READERS = {Dataset.CIFAR: partial(SequentialCIFAR10, root='cifar', train=True, download=True)}
VAL_READERS = {}
TEST_READERS = {Dataset.CIFAR: partial(SequentialCIFAR10, root='cifar', train=False, download=True)}
NUM_FEATURES = {Dataset.CIFAR: 3}
NUM_CATEGORIES = {Dataset.CIFAR: 10}


@dataclass
class LSSMTrainingArguments:
    dataset: Dataset = field(default=Dataset.CIFAR, metadata={'help': 'Dataset to train on.'})
    split: float = field(default=0.8, metadata={'help': 'Train/val split of official train dataset if no val dataset is available.'})
    batch_size: int = field(default=64, metadata={'help': 'Train batch size (eval batch size is doubled).'})

    hidden_size: int = field(default=256, metadata={'help': 'Size of hidden data representations.'})
    n_layers: int = field(default=4, metadata={'help': 'Number of LSSM layers.'})
    dropout: float = field(default=0.2, metadata={'help': 'Dropout probability.'})

    learning_rate: float = field(default=1e-3, metadata={'help': 'Learning rate for training loop.'})
    patience: int = field(default=5, metadata={'help': 'Patience before lr drop.'})
    lr_drop_factor: float = field(default=0.5, metadata={'help': 'LR drop factor when patience goes to zero.'})
    min_learning_rate: float = field(default=1e-7, metadata={'help': 'Training stops once lr goes under this value.'})
    target_metric: str = field(default='accuracy', metadata={'help': 'Set target metric for validation and choosing best model.'})

    comment: str = field(default='LSSM', metadata={'help': 'Tensorboard comment.'})
    log_examples: int = field(default=100, metadata={'help': 'Log metrics every X examples seen.'})
    verbose: bool = field(default=False, metadata={'help': 'Print validation metrics during training.'})


def compute_metrics(predictions: np.ndarray, gold_labels: np.ndarray) -> Dict[str, float]:
    return {f'accuracy': accuracy_score(gold_labels, predictions)}


def log_metrics(writer: SummaryWriter, metrics: Dict[str, float], examples_seen, *, prefix: str = '') -> None:
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(metric_name, metric_value, global_step=examples_seen)


def validate(
        model: Module,
        dataloader: DataLoader,
        loss_fn: Callable[[Tensor], Tensor],
        *,
        dataset_name: str = 'UNKNOWN'
) -> Dict[str, float]:

    model.eval()

    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        total_loss = 0

        for batch, labels in tqdm(dataloader, desc=f'Evaluating on {dataset_name} dataset'):
            logits = model(batch)
            loss = loss_fn(logits)

            total_loss += loss.item() * len(logits)

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(predictions, labels)
    metrics['loss'] = total_loss / len(predictions)

    return metrics


if __name__ == '__main__':
    args: LSSMTrainingArguments = LSSMTrainingArguments.parse_args()
    run_args = {f'--{k}': v for k, v in args.__dict__.items()}
    print('Run arguments:')
    pprint(run_args)

    # INIT DATASETS --

    train_reader = TRAIN_READERS[args.dataset]()
    test_reader = TEST_READERS[args.dataset]()

    if args.dataset not in VAL_READERS:
        train_length = int(args.split * len(train_reader))
        val_length = len(train_reader) - train_length
        train_reader, val_reader = random_split(train_reader, lengths=(train_length, val_length))
    else:
        val_reader = VAL_READERS[args.dataset]

    train_batch_size = args.batch_size
    eval_batch_size = train_batch_size * 2

    train_dataloder = DataLoader(train_reader, shuffle=True, batch_size=train_batch_size)
    val_dataloder = DataLoader(val_reader, shuffle=False, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_reader, shuffle=False, batch_size=eval_batch_size)

    # -- INIT DATASETS

    tb_writer = SummaryWriter(comment=args.comment)

    model = S4Model(
        d_input=NUM_FEATURES[args.dataset],
        d_model=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        block_class=StateSpace
    )
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    curr_lr = args.learning_rate
    n_epoch = 0
    while curr_lr > args.min_learning_rate:
        n_epoch += 1
        for batch in tqdm(train_dataloder, desc=f'Training {n_epoch}'):
            pass



