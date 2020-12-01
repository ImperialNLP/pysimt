import torch

from ..attention import ScaledDotAttention
from .base_sublayer import BaseSublayer


class ParallelMMCrossAttentionSublayer(BaseSublayer):
    def __init__(self, model_dim, n_heads, dropout=0.1, attn_dropout=0.0, is_pre_norm=False, fusion='sum'):
        """
        Creates a ParallelCrossAttentionSublayer.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.attn_txt = ScaledDotAttention(model_dim, n_heads, attn_dropout)
        self.attn_img = ScaledDotAttention(model_dim, n_heads, attn_dropout)
        self.fusion = fusion

    def forward(self, query, key_txt, value_txt, mask_txt, key_img, value_img, mask_img=None):
        """
        Performs a forward pass over the CrossAttentionSublayer.
        :param query: The query. For encoder-decoder attention, it is the output from the previous decoder layer.
        :param key_txt: The key. For encoder-decoder attention, it is the output from the encoder.
        :param value_txt: The mask. For encoder-decoder attention, it is the output from the encoder.
        :param value_img:
        :param key_img:
        :param mask_txt: The textual encoder mask.
        :param mask_img: The visual features mask.
        :return: The output of the CrossAttentionSublayer.
        """
        residual = query
        query = self.apply_pre_norm_if_needed(query)

        attn_txt, attn_weights_txt = self.attn_txt((query, key_txt, value_txt, mask_txt))
        attn_img, attn_weights_img = self.attn_img((query, key_img, value_img, mask_img))

        attn_combined = torch.add(attn_txt, attn_img)
        out = self.apply_residual(residual, attn_combined)
        out = self.apply_post_norm_if_needed(out)
        return out, {'txt': attn_weights_txt, 'img': attn_weights_img}
