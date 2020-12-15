"""A convenience feed-forward layer with non-linearity support."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.nn import get_activation_fn


class FF(nn.Module):
    """A convenience feed-forward layer with non-linearity option.

    Args:
        input_size: The size of the input features
        hidden_size: The size of the output features
        bias: If `False`, disables the bias component
        bias_zero: If `False`, randomly initialize the bias instead of zero
            initialization
        activ: The activation function name that will be searched
            in `torch` and `torch.nn.functional` modules. `None` or `linear`
            disables the activation function

    Example:
        >>> FF(300, 400, bias=True, activ='tanh') # a tanh MLP
        >>> FF(300, 400, bias=False, activ=None) # a linear layer
    """

    def __init__(self, input_size, hidden_size, bias=True,
                 bias_zero=True, activ=None):
        """"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.bias_zero = bias_zero
        self.activ_type = activ
        if self.activ_type in (None, 'linear'):
            self.activ_type = 'linear'
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.activ = get_activation_fn(activ)

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            if self.bias_zero:
                self.bias.data.zero_()
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.activ(F.linear(input, self.weight, self.bias))

    def __repr__(self):
        repr_ = self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', hidden_size=' + str(self.hidden_size) \
            + ', activ=' + str(self.activ_type) \
            + ', bias=' + str(self.use_bias)
        if self.use_bias:
            repr_ += ', bias_zero=' + str(self.bias_zero)
        return repr_ + ')'
