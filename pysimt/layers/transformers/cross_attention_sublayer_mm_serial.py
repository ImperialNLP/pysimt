from torch import nn

from .cross_attention_sublayer import CrossAttentionSublayer


class SerialMMCrossAttentionSublayer(nn.Module):
    def __init__(self, model_dim, n_heads, dropout=0.1,
                 attn_dropout=0.0, is_pre_norm=False):
        """
        Creates a ParallelCrossAttentionSublayer.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param dropout: The dropout rate for the residual connection.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__()
        self.attn_txt = CrossAttentionSublayer(
            model_dim, n_heads, dropout, attn_dropout, is_pre_norm)
        self.attn_img = CrossAttentionSublayer(
            model_dim, n_heads, dropout, attn_dropout, is_pre_norm)

    def forward(self, query, key_txt, value_txt, mask_txt,
                key_img, value_img,  mask_img=None):
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
        attn_txt, attn_weights_txt = self.attn_txt(
            query, key_txt, value_txt, mask_txt)
        attn_img, attn_weights_img = self.attn_img(
            attn_txt, key_img, value_img, mask_img)
        return attn_img, {'txt': attn_weights_txt, 'img': attn_weights_img}
