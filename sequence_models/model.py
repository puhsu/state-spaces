""" Isotropic deep sequence model backbone, in the style of ResNets / Transformers.

The SequenceModel class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation
"""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from .config import to_list, to_dict
from .block import SequenceResidualBlock
from .base import SequenceModule
from .components import Normalization, DropoutNd


class SequenceModel(SequenceModule):
    def __init__(
        self,
        d_model,           # Resize input (useful for deep models with residuals)
        n_layers=1,        # Number of layers
        transposed=False,  # Transpose inputs so each layer receives (batch, dim, length)
        dropout=0.0,       # Dropout parameter applied on every residual and every layer
        tie_dropout=False, # Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        prenorm=True,      # Pre-norm vs. post-norm
        n_repeat=1,        # Each layer is repeated n times per stage before applying pooling
        layer=None,        # Layer config, must be specified
        residual=None,     # Residual config
        norm=None,         # Normalization config (e.g. layer vs batch)
        pool=None,         # Config for pooling layer per stage
        track_norms=True,  # Log norms of each layer output
        dropinp=0.0,       # Input dropout
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms

        # Input dropout (not really used)
        dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (l+1) % n_repeat == 0 else None
            block = SequenceResidualBlock(d, l+1, prenorm=prenorm, dropout=dropout, tie_dropout=tie_dropout, transposed=transposed, layer=layer, residual=residual, norm=norm, pool=pool_cfg)
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
            else:
                self.norm = Normalization(self.d_output, transposed=self.transposed, **norm)
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """ Inputs assumed to be (batch, sequence, dim) """
        if self.transposed: inputs = rearrange(inputs, 'b ... d -> b d ...')
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms: output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms: output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None: outputs = self.norm(outputs)

        if self.transposed: outputs = rearrange(outputs, 'b d ... -> b ... d')

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [_layer.state_to_tensor(_state) for (_layer, _state) in zip(self.layers, state)]
            x = [_x for _x in x if _x is not None]
            return torch.cat( x, dim=-1)
        return fn

    def default_state(self, *batch_shape, device=None):
        return [layer.default_state(*batch_shape, device=device) for layer in self.layers]

    def step(self, x, state, **kwargs):
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)

        return x, next_states

class SequenceModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder, model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model = model

    def forward(self, x):
        # print(x.shape)
        x, *_ = self.encoder(x)
        # print(x.shape)
        x, *_ = self.model(x)
        # print(x.shape)
        x, *_ = self.decoder(x)
        # print(x.shape)
        return x


class SequenceDecoder(torch.nn.Module):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = torch.nn.Identity() if d_output is None else torch.nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = torch.nn.functional.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)
