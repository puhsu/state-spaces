""" Implements a full residual block around a black box layer

Configurable options include:
normalization position: prenorm or postnorm
normalization type: batchnorm, layernorm etc.
subsampling/pooling
residual options: feedforward, residual, affine scalars, depth-dependent scaling, etc.
"""
import copy

from torch import nn

from functools import partial
from .components import Normalization, StochasticDepth, DropoutNd
from .base import SequenceModule


class SequenceResidualBlock(SequenceModule):
    def __init__(
            self,
            d_input,
            i_layer=None,  # Only needs to be passed into certain residuals like Decay
            prenorm=True,
            dropout=0.0,
            tie_dropout=False,
            transposed=False,
            layer=None,  # Config for black box module
            residual=None,  # Config for residual function
            norm=None,  # Config for normalization layer
            pool=None,
            drop_path=0.,
    ):
        super().__init__()
        layer, residual, pool = copy.deepcopy(layer), copy.deepcopy(residual), copy.deepcopy(pool)

        self.i_layer = i_layer
        self.d_input = d_input
        if layer is not None:
            layer_class = layer['class']
            del layer['class']
            self.layer = layer_class(d_input, **layer)
        else:
            self.layer = None
        self.prenorm = prenorm
        self.transposed = transposed

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            if residual is not None:
                residual_class = residual['class']
                del residual['class']
                self.residual = residual_class(i_layer, d_input, self.layer.d_output, **residual)
            else:
                self.residual = None
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool

        if pool is not None:
            pool_class = pool['class']
            del pool['class']
            self.pool = pool_class(self.d_residual, transposed=self.transposed)
        else:
            self.pool = None

        # Dropout
        dropout_cls = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm: y = self.norm(y)

        # Black box layer
        y, state = self.layer(y, state=state, **kwargs)

        # Residual
        if self.residual is not None: y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm: y = self.norm(y)

        # Pool
        if self.pool is not None: y, _ = self.pool(y)

        return y, state

    def step(self, x, state, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)

        # Residual
        # NOTE this would not work with concat residual function (catformer)
        if self.residual is not None:
            y = self.residual(x, y, transposed=False)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state
