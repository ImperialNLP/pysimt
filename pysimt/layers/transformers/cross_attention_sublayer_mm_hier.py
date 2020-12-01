import torch

from ..attention import ScaledDotAttention
from .base_sublayer import BaseSublayer


class HierarchicalMMCrossAttentionSublayer(BaseSublayer):

    def __init__(self, model_dim, n_heads, dropout=0.1, attn_dropout=0.0, is_pre_norm=False, n_hier_heads=8):
        """
        Creates a HierarchicalMMCrossAttentionSublayer.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.attn_txt = ScaledDotAttention(model_dim, n_heads, attn_dropout)
        self.attn_img = ScaledDotAttention(model_dim, n_heads, attn_dropout)
        self.attn_hierarchical = ScaledDotAttention(model_dim, n_hier_heads, attn_dropout)

    def forward(self, query, key_txt, value_txt, mask_txt, key_img, value_img, mask_img=None):
        """
        Performs a forward pass over the HierarchicalMMCrossAttentionSublayer.
        :param query: The query. For encoder-decoder attention, it is the output from the previous decoder layer.
        :param key_txt: The key for the textual modality. For encoder-decoder attention, it is the output from the encoder.
        :param value_txt: The value for the textual modality. For encoder-decoder attention, it is the output from the encoder.
        :param mask_txt: The mask. For encoder-decoder attention, it is the output from the encoder.
        :param key_img: The key for the visual modality.
        :param value_img: The value for the visual modality.
        :param mask_img: The visual features mask.
        :return: The output of the CrossAttentionSublayer.
        """
        residual = query
        query = self.apply_pre_norm_if_needed(query)

        attn_txt, attn_weights_txt = self.attn_txt((query, key_txt, value_txt, mask_txt))
        attn_img, attn_weights_img = self.attn_img((query, key_img, value_img, mask_img))
        attn_combined, combined_attn_weights = self._fuse_contexts(query, attn_img, attn_txt)

        out = self.apply_residual(residual, attn_combined)
        attn = self.apply_post_norm_if_needed(out)
        return attn, {'txt': attn_weights_txt, 'img': attn_weights_img, 'hier': combined_attn_weights}

    def _fuse_contexts(self, query, attn_img, attn_txt, combined_mask=None):
        seq_len, batch_size, model_dim = query.shape
        combined_key_value = torch.stack((attn_txt, attn_img), dim=0).view(2, -1, model_dim)
        combined_attn, combined_attn_weights = self.attn_hierarchical((query.view(1, -1, model_dim), combined_key_value,
                                                                       combined_key_value, combined_mask))
        return combined_attn.view(seq_len, batch_size, model_dim), combined_attn_weights
