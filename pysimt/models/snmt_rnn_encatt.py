import logging

from torch import nn
from ..layers.attention import MultiheadAttention

from . import SimultaneousNMT


logger = logging.getLogger('pysimt')


class EncoderSelfAttentionSimultaneousNMT(SimultaneousNMT):
    """Simultaneous self-attentive MMT i.e. the ENC-O* model in the paper."""

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'n_heads': 1,
            'att_dropout': 0.0,
        })

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        encoders = {}
        for key in self.topology.srcs.keys():
            encoders[key] = getattr(self, f'create_{key}_encoder')()

        # Separate out visual encoder to avoid multimodal decoder-side
        # attention to be enabled
        self.ff_vis_enc = encoders.pop('image')

        self.encoders = nn.ModuleDict(encoders)
        self.dec = self.create_decoder(encoders=self.encoders)

        # create the cross-modal self-attention network
        self.mm_attn = MultiheadAttention(
            self.opts.model['enc_dim'], self.opts.model['enc_dim'],
            n_heads=self.opts.model['n_heads'],
            dropout=self.opts.model['att_dropout'], attn_type='cross')
        self.mm_lnorm = nn.LayerNorm(self.opts.model['enc_dim'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.encoders[str(self.sl)].emb.weight = self.dec.emb.weight

    def cache_enc_states(self, batch):
        """Caches encoder states internally by forward-pass'ing each encoder."""
        self.encoders['src'](batch['src'])
        self.ff_vis_enc(batch['image'])

        src_states, src_mask = self.encoders['src'].get_states()
        img_states, img_mask = self.ff_vis_enc.get_states()

        # key values are image states
        kv = img_states.transpose(0, 1)
        attn_out = self.mm_attn(
            q=src_states.transpose(0, 1), k=kv, v=kv,
            q_mask=src_mask.transpose(0, 1).logical_not()).transpose(0, 1)

        # Inject this into the encoder itself for caching
        self.encoders['src']._states = self.mm_lnorm(src_states + attn_out)
