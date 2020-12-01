from ..attention import ScaledDotAttention
from .base_sublayer import BaseSublayer


class CrossAttentionSublayer(BaseSublayer):
    def __init__(self, model_dim, n_heads, dropout=0.1,
                 attn_dropout=0.0, is_pre_norm=False):
        """
        Creates a CrossAttentionSublayer.
        :param model_dim: The model dimension.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.attn = ScaledDotAttention(model_dim, n_heads, attn_dropout)

    def forward(self, query, key, value, mask=None, **kwargs):
        """
        Performs a forward pass over the CrossAttentionSublayer.
        :param query: The query. For encoder-decoder attention, it is the output from the previous decoder layer.
        :param key: The key. For encoder-decoder attention, it is the output from the encoder.
        :param value: The mask. For encoder-decoder attention, it is the output from the encoder.
        :param mask: The mask. For encoder-decoder attention, it is the encoder mask.
        :return: The output of the CrossAttentionSublayer.
        """
        residual = query
        query = self.apply_pre_norm_if_needed(query)
        attn_out, attn_weights = self.attn((query, key, value, mask))
        out = self.apply_residual(residual, attn_out)
        out = self.apply_post_norm_if_needed(out)
        return out, attn_weights
