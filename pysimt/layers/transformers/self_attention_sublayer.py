from ..attention import ScaledDotAttention
from .base_sublayer import BaseSublayer


class SelfAttentionSublayer(BaseSublayer):

    def __init__(self, model_dim, n_heads, dropout=0.1,
                 attn_dropout=0.0, is_pre_norm=False):
        """
        Creates a SelfAttentionSublayer.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.attn = ScaledDotAttention(model_dim, n_heads, attn_dropout)

    def forward(self, x, mask=None):
        """
        Performs a forward pass over the SelfAttentionSublayer.
        :param x: The input. Will be used as query, key and value.
        :param mask: The input mask.
        :return: The output of the SelfAttentionSublayer.
        """
        residual = x
        x = self.apply_pre_norm_if_needed(x)
        attn_out, attn_weights = self.attn((x, x, x, mask))
        out = self.apply_residual(residual, attn_out)
        out = self.apply_post_norm_if_needed(out)
        return out, attn_weights
