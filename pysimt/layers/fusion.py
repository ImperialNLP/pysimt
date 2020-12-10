import operator
from typing import Optional
from functools import reduce

import torch

from . import FF
from ..utils.nn import get_activation_fn


class Fusion(torch.nn.Module):
    """A convenience layer that merges an arbitrary number of inputs using
    concatenation, addition or multiplication. It then applies an optional
    non-linearity given by the `activ` argument. If `operation==concat`,
    additional arguments should be provided to define an adaptor MLP
    that will project the concatenated vector into a lower dimensional space.

    Args:
        operation: `concat`, `sum` or `mul` for concatenation, addition, and
            multiplication respectively
        activ: The activation function name that will be searched
            in `torch` and `torch.nn.functional` modules. `None` or `linear`
            disables the activation function
        input_size: Only required for `concat` fusion, to denote the concatenated
            input vector size. This will be used to add an MLP adaptor layer
            after concatenation to project the fused vector into a lower
            dimension
        output_size: Only required for `concat` fusion, to denote the
            output size of the aforementioned adaptor layer
    """
    def __init__(self,
                 operation: str = 'concat',
                 activ: Optional[str] = 'linear',
                 input_size: Optional[int] = None,
                 output_size: Optional[int] = None):
        """"""
        super().__init__()

        self.operation = operation
        self.activ = activ
        self.forward = getattr(self, '_{}'.format(self.operation))
        self.activ = get_activation_fn(activ)
        self.adaptor = lambda x: x

        if self.operation == 'concat' or input_size != output_size:
            self.adaptor = FF(input_size, output_size, bias=False, activ=None)

    def _sum(self, inputs):
        return self.activ(self.adaptor(reduce(operator.add, inputs)))

    def _mul(self, inputs):
        return self.activ(self.adaptor(reduce(operator.mul, inputs)))

    def _concat(self, inputs):
        return self.activ(self.adaptor(torch.cat(inputs, dim=-1)))

    def __repr__(self):
        return f"Fusion(type={self.operation}, activ={self.activ})"
