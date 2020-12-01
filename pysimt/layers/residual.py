from torch import nn


class Residual(nn.Module):
    def __init__(self, dropout=0.1):
        """
        Creates a Residual layer that computes `x + dropout(f(x))`.

        :param dropout: The dropout rate.
        """
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Performs a forward pass.

        :param inputs: The inputs, should be the residual x and f_x in a tuple.
        :return: The output of the layer.
        """
        # Unpack into `x` and `Sublayer(x)`
        x, f_x = inputs
        return x + self.dropout_layer(f_x)
