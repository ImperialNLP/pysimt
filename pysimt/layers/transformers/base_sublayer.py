from torch import nn


class BaseSublayer(nn.Module):

    def __init__(self, model_dim, dropout=0.1, is_pre_norm=False):
        """
        Creates a BaseSublayer.
        :param model_dim: The model dimension.
        :param dropout: The dropout layer.
        :param is_pre_norm: Whether it should use pre_norm transformer layers. Default: False.
        """
        super().__init__()
        self.is_pre_norm = is_pre_norm
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, **kwargs):
        raise NotImplementedError("BaseSublayer does not implement forward.")

    def apply_pre_norm_if_needed(self, x):
        """
        Applies pre_norm to the input if needed. If pre_norm is false, the input remains unchanged.
        :param x: The input.
        :return: The output.
        """
        if self.is_pre_norm:
            x = self.layer_norm(x)
        return x

    def apply_post_norm_if_needed(self, x):
        """
        Applies post_norm to the input if needed. If pre_norm is true, the input remains unchanged.
        :param x: The input.
        :return: The output.
        """
        if not self.is_pre_norm:
            x = self.layer_norm(x)
        return x

    def apply_residual(self, residual, x):
        """
        Applies the residual connection.
        :param residual: The residual.
        :param x: The input x.
        :return: The output of the residual connection.
        """
        return residual + self.dropout(x)
