from torch import nn

from . import FF
from .transformers import BaseSublayer


class PositionwiseFF(nn.Module):
    """Positionwise Feed-forward layer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, activ='gelu', dropout=0.1):
        """
        Creates a PositionwiseFF.
        :param model_dim: The model dimensions.
        :param ff_dim: The feedforward dimensions.
        :param activ: The activation function. Default: gelu
        :param dropout: The amount of dropout. Default: 0.1
        """
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.activ = activ

        # Create the layers
        self.layers = nn.Sequential(
            FF(self.model_dim, self.ff_dim, activ=self.activ),
            nn.Dropout(dropout),
            FF(self.ff_dim, self.model_dim, activ=None),
        )

    def forward(self, x):
        return self.layers(x)


class PositionwiseSublayer(BaseSublayer):
    def __init__(self, model_dim, ff_dim, ff_activ='gelu', dropout=0.1, is_pre_norm=False):
        """
        Creates a PositionwiseSublayer.
        :param model_dim: The model dimensions.
        :param ff_dim: The dimensions of the feed forward network.
        :param ff_activ: The activation of the feed forward network.
        :param dropout: The dropout rate.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.feed_forward = PositionwiseFF(model_dim, ff_dim, ff_activ, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Performs a forward pass over the PositionwiseSublayer.
        :param x: The input x.
        :param mask: The input mask.
        :return: The output from the forward pass of the PositionwiseSublayer.
        """
        residual = x
        x = self.apply_pre_norm_if_needed(x)
        x = self.feed_forward(x)
        x = self.apply_residual(residual, x)
        x = self.apply_post_norm_if_needed(x)
        return x
