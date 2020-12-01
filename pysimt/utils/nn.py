import torch
import torch.nn.functional as F


def get_activation_fn(name: str):
    """Returns a callable activation function from `torch`."""
    if name in (None, 'linear'):
        return lambda x: x
    elif name in ('sigmoid', 'tanh'):
        return getattr(torch, name)
    else:
        return getattr(F, name)
