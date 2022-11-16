import logging
import pprint
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer.utils import compute_metrics
from trainer.trainer_logging import log_metrics

logger = logging.getLogger(__name__)


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class TrainingArguments:
    learning_rate: float = field(default=1e-3, metadata={'help': 'Learning rate for training loop.'})
    patience: int = field(default=5, metadata={'help': 'Patience before lr drop.'})
    lr_drop_factor: float = field(default=0.5, metadata={'help': 'LR drop factor when patience goes to zero.'})
    min_learning_rate: float = field(default=1e-7, metadata={'help': 'Training stops once lr goes under this value.'})
    target_metric: str = field(default='accuracy', metadata={'help': 'Set target metric for validation and choosing best model.'})

    comment: str = field(default='LSSM', metadata={'help': 'Tensorboard comment.'})
    log_examples: int = field(default=10000, metadata={'help': 'Log metrics every X examples seen.'})
    verbose: bool = field(default=True, metadata={'help': 'Print validation metrics during training.'})
    log_file: Path = field(default=Path('results.csv'), metadata={'help': 'File for logging run configs and final metrics.'})


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


_Model = TypeVar('_Model', bound=Module)


def train(
        model: _Model,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        args: TrainingArguments,
        tb_writer: Optional[SummaryWriter] = None
) -> _Model:
    curr_lr = args.learning_rate
    curr_patience = args.patience

    n_epoch = 0
    examples_seen = 0
    log_loss = 0
    examples_from_last_log = 0

    best_model_state = model.state_dict()
    best_metric: Optional[float] = None
    while curr_lr > args.min_learning_rate:
        n_epoch += 1
        for batch, labels, *_ in tqdm(train_dataloader, desc=f'Training {n_epoch} epoch', leave=True):
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
                metrics = validate(model, val_dataloader, loss_fn, dataset_name='dev')
                if tb_writer is not None:
                    log_metrics(tb_writer, metrics, examples_seen=examples_seen, prefix='dev')
                    log_metrics(
                        tb_writer, {'lr': curr_lr, 'loss': log_loss / examples_from_last_log},
                        examples_seen=examples_seen,
                        prefix='train',
                    )
                if args.verbose:
                    logger.info(pprint.pformat(metrics))

                log_loss = 0
                examples_from_last_log = 0

                metric = metrics[args.target_metric]
                if best_metric is None or metric > best_metric:
                    best_model_state = model.state_dict()
                    best_metric = metric
                    curr_patience = args.patience
                else:
                    curr_patience -= 1

                if curr_patience == 0:
                    # drop lr
                    model.load_state_dict(best_model_state)
                    curr_lr *= args.lr_drop_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = curr_lr
                    curr_patience = args.patience

    # EVALUATION

    model.load_state_dict(best_model_state)
    return model
