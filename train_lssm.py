import json
import logging
import pprint
from dataclasses import field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple, Any, Type

import numpy as np
import torch
from argparse_dataclass import dataclass
from sklearn.metrics import accuracy_score
from torch import Tensor, LongTensor
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.s4 import S4
from ss_datasets import SequentialCIFAR10
from model.s4_model import S4Model
from model.lssl import StateSpace as LSSL
from ss_datasets.lra.configure import configure_lra
from ss_datasets.lra.loader import PathFinder


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


class Dataset(str, Enum):
    CIFAR = 'cifar'
    PATH64 = 'path64'
    PATH128 = 'path128'
    PATH256 = 'path256'


class PathXHandler:

    path64: Optional[PathFinder] = None
    path128: Optional[PathFinder] = None
    path256: Optional[PathFinder] = None

    __init__ = None

    @classmethod
    def get_path64(cls) -> PathFinder:
        if cls.path64 is None:
            cls.path64 = configure_lra(x=64)
        return cls.path64

    @classmethod
    def get_path128(cls) -> PathFinder:
        if cls.path128 is None:
            cls.path128 = configure_lra(x=128)
        return cls.path128

    @classmethod
    def get_path256(cls) -> PathFinder:
        if cls.path256 is None:
            cls.path256 = configure_lra(x=256)
        return cls.path256


TRAIN_READERS = {
    Dataset.CIFAR: partial(SequentialCIFAR10, root='cifar', train=True, download=True),
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_train,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_train,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_train
}
VAL_READERS = {
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_val,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_val,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_val
}
TEST_READERS = {
    Dataset.CIFAR: partial(SequentialCIFAR10, root='cifar', train=False, download=True),
    Dataset.PATH64: lambda: PathXHandler.get_path64().dataset_test,
    Dataset.PATH128: lambda: PathXHandler.get_path128().dataset_test,
    Dataset.PATH256: lambda: PathXHandler.get_path256().dataset_test
}
NUM_FEATURES = {
    Dataset.CIFAR: 3,
    Dataset.PATH64: 1,
    Dataset.PATH128: 1,
    Dataset.PATH256: 1
}
NUM_CATEGORIES = {
    Dataset.CIFAR: 10,
    Dataset.PATH64: 2,
    Dataset.PATH128: 2,
    Dataset.PATH256: 2
}


@dataclass
class LSSMTrainingArguments:
    dataset: Dataset = field(default=Dataset.CIFAR, metadata={'help': 'Dataset to train on.'})
    split: float = field(default=0.8, metadata={'help': 'Train/val split of official train dataset if no val dataset is available.'})
    batch_size: int = field(default=256, metadata={'help': 'Train batch size (eval batch size is doubled).'})

    sequence_module: str = field(default='lssl', metadata={'help': 'Sequence module block type (lssl or s4)'})
    hidden_size: int = field(default=256, metadata={'help': 'Size of hidden data representations.'})
    n_layers: int = field(default=4, metadata={'help': 'Number of LSSL layers.'})
    dropout: float = field(default=0.2, metadata={'help': 'Dropout probability.'})
    channels: int = field(default=4, metadata={'help': 'Number of channels for LSSL layers.'})

    learning_rate: float = field(default=1e-3, metadata={'help': 'Learning rate for training loop.'})
    patience: int = field(default=5, metadata={'help': 'Patience before lr drop.'})
    lr_drop_factor: float = field(default=0.5, metadata={'help': 'LR drop factor when patience goes to zero.'})
    min_learning_rate: float = field(default=1e-7, metadata={'help': 'Training stops once lr goes under this value.'})
    target_metric: str = field(default='accuracy', metadata={'help': 'Set target metric for validation and choosing best model.'})

    comment: str = field(default='LSSM', metadata={'help': 'Tensorboard comment.'})
    log_examples: int = field(default=10000, metadata={'help': 'Log metrics every X examples seen.'})
    verbose: bool = field(default=True, metadata={'help': 'Print validation metrics during training.'})
    log_file: Path = field(default=Path('results.csv'), metadata={'help': 'File for logging run configs and final metrics.'})


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Training on {DEVICE}')


def compute_metrics(predictions: np.ndarray, gold_labels: np.ndarray) -> Dict[str, float]:
    return {f'accuracy': accuracy_score(gold_labels, predictions)}


def log_metrics(writer: SummaryWriter, metrics: Dict[str, float], examples_seen, *, prefix: str = '', verbose: bool = False) -> None:
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{prefix}_{metric_name}', metric_value, global_step=examples_seen)
        if verbose:
            logger.info(f'{prefix}_{metric_name}: {metric_value}')


def validate(
        model: Module,
        dataloader: DataLoader,
        loss_fn: Callable[[Tensor, LongTensor], Tensor],
        *,
        dataset_name: str = 'UNKNOWN'
) -> Dict[str, float]:

    model.eval()

    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        total_loss = 0

        for batch, labels, *_ in tqdm(dataloader, desc=f'Evaluating on {dataset_name} dataset', leave=False):
            logits = model(batch.to(DEVICE))
            loss = loss_fn(logits, labels.to(DEVICE))

            total_loss += loss.item() * len(logits)

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(predictions, labels)
    metrics['loss'] = total_loss / len(predictions)

    return metrics


def init_block_params(args: LSSMTrainingArguments) -> Tuple[Type[Module], Dict[str, Any]]:
    if args.sequence_module == 'lssl':
        return LSSL, {
            'channels': args.channels
        }
    elif args.sequence_module == 's4':
        return S4, {
            'channels': args.channels
        }
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args: LSSMTrainingArguments = LSSMTrainingArguments.parse_args()
    run_args = {f'--{k}': str(v) for k, v in args.__dict__.items()}
    logger.info(f'Run arguments:\n{pprint.pformat(run_args)}')

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

    block_class, block_args = init_block_params(args)

    model = S4Model(
        d_input=NUM_FEATURES[args.dataset],
        d_model=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        block_class=block_class,
        block_kwargs=block_args
    ).to(DEVICE)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss()

    logger.info('Optimized model:')
    logger.info(model)

    curr_lr = args.learning_rate
    patience = args.patience

    n_epoch = 0
    examples_seen = 0
    log_loss = 0
    examples_from_last_log = 0

    best_model_state = model.state_dict()
    best_metric: Optional[float] = None
    while curr_lr > args.min_learning_rate:
        n_epoch += 1
        for batch, labels, *_ in tqdm(train_dataloder, desc=f'Training {n_epoch} epoch', leave=True):
            model.train()

            optimizer.zero_grad()
            logits = model(batch.to(DEVICE))
            loss = loss_fn(logits, labels.to(DEVICE))

            loss.backward()
            optimizer.step()

            examples_seen += len(batch)

            log_loss += loss.item() * len(batch)
            examples_from_last_log += len(batch)

            if examples_from_last_log > args.log_examples:
                metrics = validate(model, val_dataloder, loss_fn, dataset_name='dev')
                log_metrics(tb_writer, metrics, examples_seen=examples_seen, prefix='dev', verbose=args.verbose)
                log_metrics(
                    tb_writer, {'lr': curr_lr, 'loss': log_loss / examples_from_last_log},
                    examples_seen=examples_seen,
                    prefix='train',
                    verbose=args.verbose
                )

                log_loss = 0
                examples_from_last_log = 0

                metric = metrics[args.target_metric]
                if best_metric is None or metric > best_metric:
                    best_model_state = model.state_dict()
                    best_metric = metric
                    patience = args.patience
                else:
                    patience -= 1

                if patience == 0:
                    # drop lr
                    model.load_state_dict(best_model_state)
                    curr_lr *= args.lr_drop_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = curr_lr
                    patience = args.patience

    # EVALUATION

    model.load_state_dict(best_model_state)

    val_metrics = validate(model, val_dataloder, loss_fn, dataset_name='dev')
    test_metrics = validate(model, test_dataloader, loss_fn, dataset_name='test')

    with open(args.log_file, 'a') as f:
        f.write(f'{json.dumps(run_args)},{json.dumps(val_metrics)},{json.dumps(test_metrics)},'
                f'{val_metrics[args.target_metric]},{test_metrics[args.target_metric]}\n')

