import logging
from trainer.trainer_logging import setup_logging, dump_metrics

setup_logging()
logger = logging.getLogger(__name__)

import pprint
from dataclasses import field, dataclass
from typing import Dict, Tuple, Any, Type

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser

from model.s4 import S4
from model.s4_model import S4Model
from model.lssl import StateSpace as LSSL
from trainer.dataset import NUM_FEATURES, get_dataloaders, DatasetArguments
from trainer.loops import validate, train, TrainingArguments


@dataclass
class LSSMTrainingArguments:
    sequence_module: str = field(default='lssl', metadata={'help': 'Sequence module block type (lssl or s4)'})
    hidden_size: int = field(default=256, metadata={'help': 'Size of hidden data representations.'})
    n_layers: int = field(default=4, metadata={'help': 'Number of LSSL layers.'})
    dropout: float = field(default=0.2, metadata={'help': 'Dropout probability.'})
    channels: int = field(default=4, metadata={'help': 'Number of channels for LSSL layers.'})


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Training on {DEVICE}')


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
    parser = HfArgumentParser(dataclass_types=[DatasetArguments, TrainingArguments, LSSMTrainingArguments])
    data_args, train_args, model_args = parser.parse_args_into_dataclasses()

    data_args: DatasetArguments
    train_args: TrainingArguments
    model_args: LSSMTrainingArguments

    run_args = {f'--{k}': str(v) for k, v in parser.parse_args().__dict__.items()}
    logger.info(f'Run arguments:\n{pprint.pformat(run_args)}')

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_args)

    tb_writer = SummaryWriter(comment=train_args.comment)

    block_class, block_args = init_block_params(model_args)

    model = S4Model(
        d_input=NUM_FEATURES[data_args.dataset],
        d_model=model_args.hidden_size,
        n_layers=model_args.n_layers,
        dropout=model_args.dropout,
        block_class=block_class,
        block_kwargs=block_args
    ).to(DEVICE)
    optimizer = AdamW(params=model.parameters(), lr=train_args.learning_rate)
    loss_fn = CrossEntropyLoss()

    model = train(model, loss_fn, optimizer, train_dataloader, val_dataloader, train_args, tb_writer=tb_writer)

    val_metrics = validate(model, val_dataloader, loss_fn, dataset_name='dev')
    test_metrics = validate(model, test_dataloader, loss_fn, dataset_name='test')

    dump_metrics(train_args.log_file, run_args, val_metrics, test_metrics, train_args.target_metric)

