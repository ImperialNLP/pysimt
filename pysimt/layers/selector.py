from torch import nn, Tensor
from typing import Iterable, Any


class Selector(nn.Module):
    """Utility layer that selects and returns a particular element out of
    a tuple. It is useful to select a particular output from the previous layer,
    when used in constructs such as `torch.nn.Sequential()`.

    Args:
        index: The position to select from the given input.

    Example:
        >>> layers = []
        >>> layers.append(torch.nn.GRU(200, 400))
        # By default, GRU returns (output, h_n) but we are not interested in h_n
        >>> layers.append(Selector(0))
        >>> layers.append(torch.nn.Dropout(0.2))
        >>> self.block = nn.Sequential(*layers)
    """
    def __init__(self, index: int):
        """"""
        super().__init__()
        self.index = index

    def forward(self, x: Iterable[Tensor]) -> Tensor:
        """Returns the pre-determined `self.index`'th position of `x`."""
        return x[self.index]

    def __repr__(self):
        return f"Selector(index={self.index})"
