import torch

from .base_sublayer import BaseSublayer
from ..attention import ScaledDotAttention
from ...utils.nn import generate_default_mask


class FlatMMCrossAttentionSublayer(BaseSublayer):
    def __init__(self, model_dim, n_heads, dropout=0.1,
                 attn_dropout=0.0, is_pre_norm=False):
        """
        Creates a FlatMMCrossAttentionSublayer.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.multimodal_attn = ScaledDotAttention(
            model_dim, n_heads, attn_dropout)

    def forward(self, query, key_txt, value_txt, mask_txt,
                key_img, value_img, mask_img=None):
        """
        Performs a forward pass.
        :param query: The query for the attention layers.
        :param key_txt: The key for the textual modality. If None, it is set to the query.
        :param value_txt: The value for the textual modality. If None, it is set to the query.
        :param mask_txt: The textual modality mask.
        :param key_img: The key for the visual modality.
        :param value_img: The value for the visual modality.
        :param mask_img: The visual modality mask. Default: None.
        :return:
        """
        residual = query
        query = self.apply_pre_norm_if_needed(query)
        if key_txt is None:
            key_txt = query
        if value_txt is None:
            value_txt = query

        combined_mask = self._generate_combined_mask(
            key_img, mask_img, mask_txt)

        multimodal_key = torch.cat((key_img, key_txt), dim=0)
        multimodal_value = torch.cat((value_img, value_txt), dim=0)
        attn_multimodal, attn_weights = self.multimodal_attn(
            (query, multimodal_key, multimodal_value, combined_mask))

        out = self.apply_residual(residual, attn_multimodal)
        out = self.apply_post_norm_if_needed(out)
        return out, attn_weights

    @staticmethod
    def _generate_combined_mask(key_img, mask_img, mask_txt):
        if mask_img is None:
            mask_img = generate_default_mask(key_img, mask_txt.shape[1])
        combined_mask = torch.cat((mask_img, mask_txt), dim=-1)
        return combined_mask
