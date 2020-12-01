from torch import nn

from .. import TFEmbedding
from ..transformers import FlatMMCrossAttentionSublayer
from ..transformers import SelfAttentionSublayer
from ..positionwise_ff import PositionwiseSublayer
from ...utils.mask_utils import generate_padding_masks, generate_lookahead_mask


class TFEncoderBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, n_heads, ff_activ='gelu',
                 dropout=0.1, attn_dropout=0.0,
                 pre_norm=True, enc_img_attn=None):
        """
        Creates a TFEncoderBlock, consisting of a self-attention sublayer and a position-wise feed forward sublayer.
        :param model_dim: The model dimensions.
        :param ff_dim: The feed forward network dimensions.
        :param n_heads: The number of attention heads.
        :param ff_activ: The feed forward network activation function.
        :param dropout: The dropout.
        :param pre_norm: Whether it should use 'pre_norm' layer types or 'post_norm' Default True.
        :param enc_img_attn: The encoder image attention. Possible values: ['flat', 'None']. Default: None.
        """
        super().__init__()
        self.enc_img_attn = enc_img_attn
        if enc_img_attn == 'flat':
            self.multimodal_attn = FlatMMCrossAttentionSublayer(
                model_dim, n_heads, dropout, attn_dropout, pre_norm)
        else:
            self.self_attn = SelfAttentionSublayer(
                model_dim, n_heads, dropout, attn_dropout, pre_norm)
        self.feed_forward = PositionwiseSublayer(
                model_dim, ff_dim, ff_activ, dropout, pre_norm)

    def forward(self, encoder_x, encoder_mask=None, image_x=None, image_mask=None):
        """
        Performs a forward pass of an encoder block.
        :param encoder_x: The encoder's source text input.
        :param encoder_mask: The encoder's source text input mask.
        :param image_x: The encoder's image input.
        :param image_mask: The encoder's image input mask.

        :return: The output of the forward pass.
        """
        if self.enc_img_attn == 'flat' and image_x is not None:
            encoder_x, attn_weights = self.multimodal_attn(
                encoder_x, key_txt=None, value_txt=None, mask_txt=encoder_mask,
                key_img=image_x, value_img=image_x, mask_img=image_mask)
            all_attn_weights = {'multimodal': attn_weights}
        else:
            encoder_x, attn_weights = self.self_attn(encoder_x, encoder_mask)
            all_attn_weights = {'self': attn_weights}

        return self.feed_forward(encoder_x, encoder_mask), all_attn_weights


class TFEncoder(nn.Module):
    """Encoder block for Transformer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, n_heads, n_layers, num_embeddings,
                 ff_activ='gelu', dropout=0.1, attn_dropout=0.0, pre_norm=True,
                 enc_bidirectional=False, enc_img_attn=None,
                 store_attn_weights=False):
        """
        Creates a TFEncoder.
        :param model_dim: The model dimension.
        :param ff_dim: The feed-forward layer dimension.
        :param n_heads: The number of heads.
        :param n_layers: The number of layers.
        :param num_embeddings: The number of the embeddings.
        :param ff_activ: The feed forward layer activation function. Default 'gelu'.
        :param dropout: The dropout value. Default 0.1.
        :param pre_norm: Whether it should use 'pre_norm' layer types or 'post_norm' Default True.
        :param enc_bidirectional: The encoder should be bidirectional. Default: False
        """
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_activ = ff_activ
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.enc_bidirectional = enc_bidirectional
        self.enc_img_attn = enc_img_attn
        self.store_attn_weights = store_attn_weights

        self.src_embedding = TFEmbedding(
            num_embeddings=num_embeddings, embedding_dim=self.model_dim, dropout=dropout)

        self._encoder_mask = None
        self._encoder_states = None
        self.final_layer_norm = None
        self._all_attention_weights = []

        blocks = []
        for _ in range(self.n_layers):
            layers = TFEncoderBlock(
                model_dim=self.model_dim, n_heads=self.n_heads, ff_dim=self.ff_dim,
                ff_activ=self.ff_activ, dropout=self.dropout, attn_dropout=attn_dropout,
                pre_norm=self.pre_norm, enc_img_attn=enc_img_attn)
            blocks.append(layers)

        self.blocks = nn.ModuleList(blocks)

        if self.pre_norm:
            self.final_layer_norm = nn.LayerNorm(self.model_dim, eps=1e-6)

    def forward(self, x, **kwargs):
        """Forward-pass of the encoder block.

        :param x: input tensor, shape (s_len, bsize, model_dim)
        :return: The output after applying the forward pass and the mask.
        """
        padding_mask = generate_padding_masks(x)
        mask = padding_mask
        if not self.enc_bidirectional:
            mask = mask | generate_lookahead_mask(x)

        x = self.src_embedding(x)
        image, image_mask = self._get_image_data(kwargs)

        self._all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask, image_x=image, image_mask=image_mask)
            if self.store_attn_weights:
                self._all_attention_weights.append(attn_weights)

        if self.pre_norm:
            x = self.final_layer_norm(x)

        self._encoder_states = x
        self._encoder_mask = padding_mask
        return self._encoder_states, self._encoder_mask

    def get_attention_weights(self):
        return self._all_attention_weights

    def get_states(self, up_to=int(1e6)):
        """Reveals partial source information through `up_to` argument.
        Useful for simultaneous NMT encodings."""
        if not self.enc_bidirectional:
            assert self._encoder_states is not None, "Call encoder first to cache states!"
            return self._encoder_states[:up_to], self._encoder_mask[:, :, :up_to]
        else:
            raise NotImplementedError(
                "get_states is not implemented for bidirectional encoders as the states cannot be easily cached")

    @staticmethod
    def _get_image_data(kwargs):
        image_x = None
        image_mask = None
        if 'img_data' in kwargs:
            image_x, image_mask = kwargs['img_data']
        return image_x, image_mask
