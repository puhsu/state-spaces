import logging

from custom_attention.multihead import Attention
from trainer.trainer_logging import setup_logging, dump_metrics

setup_logging()
logger = logging.getLogger(__name__)

import pprint
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import Transformer, CrossEntropyLoss, Module, Linear, Parameter, init, TransformerEncoderLayer, Sequential, ReLU
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser

from trainer.dataset import DatasetArguments, get_dataloaders, NUM_FEATURES, NUM_CATEGORIES, Dataset
from trainer.loops import TrainingArguments, train, validate


@dataclass
class TransformerTrainingArguments:
    num_layers: int = field(default=12, metadata={'help': 'Number of layers in transformer model.'})
    num_heads: int = field(default=8, metadata={'help': 'Number of attention heads per layer.'})
    hidden_size: int = field(default=512, metadata={'help': 'Transformer hidden dims.'})
    feedforward_size: int = field(default=2048, metadata={'help': 'Size of feedforward network in encoder.'})
    dropout: float = field(default=0.1, metadata={'help': 'Attention dropout.'})


class FlashTransformerForClassification(Module):

    def __init__(self, transformer_args: TransformerTrainingArguments, num_features: int, num_categories: int):
        super().__init__()

        self._features_transition = Linear(num_features, transformer_args.hidden_size)
        self._transformer = Transformer(
            d_model=transformer_args.hidden_size,
            custom_decoder=lambda target, memory, *_, **__: memory,  # no decoder
            nhead=transformer_args.num_heads,
            num_encoder_layers=transformer_args.num_layers,
            dim_feedforward=transformer_args.feedforward_size,
            dropout=transformer_args.dropout,
            batch_first=True
        )
        for layer in self._transformer.encoder.layers:
            # swap for flash attention
            layer: TransformerEncoderLayer
            layer.self_attn = Attention(transformer_args.hidden_size, transformer_args.num_heads, dropout=transformer_args.dropout)
        self._head = Sequential(
            Linear(transformer_args.hidden_size, transformer_args.feedforward_size),
            ReLU(inplace=True),
            Linear(transformer_args.feedforward_size, num_categories),
        )

    def forward(self, features: Tensor) -> Tensor:
        transformer_input = self._features_transition(features)
        transformer_output = self._transformer(transformer_input, transformer_input)
        return self._head(transformer_output.mean(dim=1))


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Training on {DEVICE}')


if __name__ == '__main__':
    parser = HfArgumentParser(dataclass_types=[DatasetArguments, TrainingArguments, TransformerTrainingArguments])
    data_args, train_args, model_args = parser.parse_args_into_dataclasses()

    data_args: DatasetArguments
    train_args: TrainingArguments
    model_args: TransformerTrainingArguments

    run_args = {f'--{k}': str(v) for k, v in parser.parse_args().__dict__.items()}
    logger.info(f'Run arguments:\n{pprint.pformat(run_args)}')

    data_args.dataset = Dataset(data_args.dataset)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_args)

    tb_writer = SummaryWriter(comment=train_args.comment)

    model = FlashTransformerForClassification(model_args, NUM_FEATURES[data_args.dataset], NUM_CATEGORIES[data_args.dataset]).to(DEVICE)
    optimizer = AdamW(params=model.parameters(), lr=train_args.learning_rate)
    loss_fn = CrossEntropyLoss()

    model = train(model, loss_fn, optimizer, train_dataloader, val_dataloader, train_args, tb_writer=tb_writer)

    val_metrics = validate(model, val_dataloader, loss_fn, dataset_name='dev')
    test_metrics = validate(model, test_dataloader, loss_fn, dataset_name='test')

    dump_metrics(train_args.log_file, run_args, val_metrics, test_metrics, train_args.target_metric)