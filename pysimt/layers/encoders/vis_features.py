from torch import nn

from ...utils.mask_utils import generate_visual_features_padding_masks
from .. import FF


class VisualFeaturesEncoder(nn.Module):
    """A facility encoder for pre-extracted visual features.

    Arguments:
        input_size (int): number of channels in the last dimension of
            the features.
        proj_dim(int, optional): If not `None`, add a final projection
            layer similar to a 1x1 Conv2D.
        proj_activ(str, optional): Non-linearity for projection layer.
            `None` or `linear` does not apply any non-linearity.
        layer_norm(bool, optional): Apply layer normalization.
        l2_norm(bool, optional): L2-normalize features.
        dropout (float, optional): Optional dropout to be applied on the
            projected visual features.
        pool (bool, optional): If True, applies global average pooling
            to reduce conv features to a single vector.

    Input:
        x (Tensor): A tensor of shape (w*h, batch_size, input_size)

    Output:
        h (Tensor): A tensor of shape (w*h, batch_size, proj_dim)
        mask (None): No masking is done for visual features.
    """
    def __init__(self, input_size, proj_dim=None, proj_activ=None,
                 layer_norm=False, l2_norm=False, dropout=0.0, pool=False, image_masking=False):
        super().__init__()

        self.ctx_size = input_size
        self.l2_norm = l2_norm
        self._image_masking = image_masking

        output_layers = []
        if proj_dim is not None:
            output_layers.append(
                FF(input_size, proj_dim, activ=proj_activ))
            self.ctx_size = proj_dim

        if layer_norm:
            output_layers.append(nn.LayerNorm(self.ctx_size))

        if dropout > 0:
            output_layers.append(nn.Dropout(dropout))

        self.output = nn.Sequential(*output_layers)

        # Variables for caching
        self._states, self._mask = None, None

    def forward(self, x, **kwargs):
        if self._image_masking:
            self._mask = generate_visual_features_padding_masks(x)
        if self.l2_norm:
            x.div_(x.norm(p=2, dim=-1, keepdim=True))
        self._states = self.output(x)
        return self._states, self._mask

    def get_states(self, up_to=int(1e6)):
        assert self._states is not None, \
            "encoder was not called for caching the states."
        return self._states, self._mask
