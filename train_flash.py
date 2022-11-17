import logging
import math

from torch.nn.init import normal_, uniform_
from torch_position_embedding import PositionEmbedding

from custom_attention.multihead import Attention
from trainer.trainer_logging import setup_logging, dump_metrics

setup_logging()
logger = logging.getLogger(__name__)

import pprint
from dataclasses import dataclass, field

import torch
from torch import Tensor, LongTensor, zeros, arange
from torch.nn import Transformer, CrossEntropyLoss, Module, Linear, TransformerEncoderLayer, Sequential, ReLU, LayerNorm, \
    Embedding, Parameter, Dropout
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser

from trainer.dataset import DatasetArguments, get_dataloaders, NUM_CATEGORIES, Dataset, VOCAB_SIZE, set_tokenize, MAX_LENGTH
from trainer.loops import TrainingArguments, train, validate


@dataclass
class TransformerTrainingArguments:
    pool: str = field(default='mean', metadata={'help': 'mean or cls'})
    embedding_dim: int = field(default=128, metadata={'help': 'Embedding dims'})
    freeze_positional_embedding: bool = field(default=False, metadata={'help': 'Freeze pos'})
    num_layers: int = field(default=12, metadata={'help': 'Number of layers in transformer model.'})
    num_heads: int = field(default=8, metadata={'help': 'Number of attention heads per layer.'})
    hidden_size: int = field(default=512, metadata={'help': 'Transformer hidden dims.'})
    feedforward_size: int = field(default=2048, metadata={'help': 'Size of feedforward network in encoder.'})
    dropout: float = field(default=0.3, metadata={'help': 'Embedding dropout.'})
    attention_dropout: float = field(default=0.2, metadata={'help': 'Attention dropout'})


set_tokenize(True)


class PositionalEncoding(Module):
    def __init__(self, embed_dim: int, max_len: int, requires_grad: bool):
        super().__init__()
        pe = zeros(max_len, embed_dim)
        position = arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = Parameter(pe, requires_grad=requires_grad)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FlashTransformerForClassification(Module):

    def __init__(self, transformer_args: TransformerTrainingArguments, num_categories: int, vocab_size: int, input_length: int):
        super().__init__()

        self._pool_mode = transformer_args.pool
        self._d_model = transformer_args.hidden_size

        if self._pool_mode == 'cls':
            self._cls_embedding = Parameter(torch.empty(transformer_args.embedding_dim))
            uniform_(self._cls_embedding.data, -0.1, 0.1)

        self._input_embed = Embedding(vocab_size, transformer_args.embedding_dim)
        uniform_(self._input_embed.weight, -0.1, 0.1)

        self._pos_embed = PositionalEncoding(
            max_len=input_length,
            embed_dim=transformer_args.embedding_dim,
            requires_grad=not transformer_args.freeze_positional_embedding
        )
        self._embed_transition = Linear(transformer_args.embedding_dim, transformer_args.hidden_size)
        self._embed_dropout = Dropout(transformer_args.dropout)

        self._transformer = Transformer(
            d_model=transformer_args.hidden_size,
            custom_decoder=lambda target, memory, *_, **__: memory,  # no decoder
            nhead=transformer_args.num_heads,
            num_encoder_layers=transformer_args.num_layers,
            dim_feedforward=transformer_args.feedforward_size,
            dropout=transformer_args.dropout,
            batch_first=True,
            norm_first=True
        )
        for layer in self._transformer.encoder.layers:
            # swap for flash attention
            layer: TransformerEncoderLayer
            layer.self_attn = Attention(
                transformer_args.hidden_size, transformer_args.num_heads,
                dropout=transformer_args.attention_dropout
            )

        self._encoder_norm = LayerNorm(transformer_args.hidden_size)
        self._head = Sequential(
            Linear(transformer_args.hidden_size, transformer_args.feedforward_size),
            ReLU(inplace=True),
            Linear(transformer_args.feedforward_size, num_categories),
        )

    def forward(self, input_ids: LongTensor) -> Tensor:
        batch_size, _ = input_ids.shape
        embedded_inputs = self._pos_embed(self._input_embed(input_ids.long()) * math.sqrt(self.d_model))
        if self._pool_mode == 'cls':
            embedded_inputs = torch.cat([self._cls_embedding.view(1, 1, -1).repeat(batch_size, 1, 1), embedded_inputs], dim=1)

        embedded_inputs = self._embed_transition(self._embed_dropout(embedded_inputs))
        transformer_output = self._encoder_norm(self._transformer(embedded_inputs, embedded_inputs))

        pooled = transformer_output[:, 0] if self._pool_mode == 'cls' else transformer_output.mean(dim=1)
        return self._head(pooled)


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

    model = FlashTransformerForClassification(
        model_args,
        NUM_CATEGORIES[data_args.dataset],
        VOCAB_SIZE[data_args.dataset],
        MAX_LENGTH[data_args.dataset]
    ).to(DEVICE)
    optimizer = AdamW(params=model.parameters(), lr=train_args.learning_rate, weight_decay=0.0)
    loss_fn = CrossEntropyLoss()

    model = train(
        model, loss_fn, optimizer, train_dataloader, val_dataloader,
        args=train_args,
        tb_writer=tb_writer,
        batch_size=data_args.batch_size
    )

    val_metrics = validate(model, val_dataloader, loss_fn, dataset_name='dev')
    test_metrics = validate(model, test_dataloader, loss_fn, dataset_name='test')

    dump_metrics(train_args.log_file, run_args, val_metrics, test_metrics, train_args.target_metric)