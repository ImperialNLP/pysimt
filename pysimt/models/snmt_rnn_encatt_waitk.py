import logging

from . import EncoderSelfAttentionSimultaneousNMT

logger = logging.getLogger('pysimt')


"""This is the training-time wait-k model from:
    Ma et al. (2018), STACL: Simultaneous Translation with Implicit Anticipation
   and Controllable Latency using Prefix-to-Prefix Framework, arXiv:1810.08398

The only required parameter is the `k` argument for training. When decoding,
pass the `k` argument explicitly to `pysimt translate`. A large enough `k`
should produce the same results as the `snmt.py` model.
"""


class EncoderSelfAttentionSimultaneousWaitKNMT(EncoderSelfAttentionSimultaneousNMT):
    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            # Decoding/training simultaneous NMT args
            'translator_type': 'wk',        # This model implements train-time wait-k
            'translator_args': {'k': 1e4},  # k as in wait-k in training
            'consecutive_warmup': 0,        # consecutive training for this many epochs
        })

    def __init__(self, opts):
        super().__init__(opts)
        assert self.opts.model['translator_type'] != 'bs', \
            'Beam search not compatible with simultaneous models'

    def forward(self, batch, **kwargs):
        """Training forward-pass with explicit timestep-based loop."""
        loss = 0.0

        k = int(self.opts.model['translator_args']['k'])
        if self.training:
            epoch_count = kwargs['ectr']
            if epoch_count <= self.opts.model['consecutive_warmup']:
                # warming up, use full contexts
                k = int(1e4)

        # Cache encoder states first
        self.cache_enc_states(batch)

        # Initial state is None i.e. 0.
        h = self.dec.f_init()

        # Convert target token indices to embeddings -> T*B*E
        y = batch[self.tl]
        y_emb = self.dec.emb(y)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.size(0) - 1):
            ###########################################
            # waitk: pass partial context incrementally
            ###########################################
            state_dict = self.get_enc_state_dict(up_to=k + t)
            log_p, h = self.dec.f_next(state_dict, y_emb[t], h)
            loss += self.dec.nll_loss(log_p, y[t + 1])

        return {
            'loss': loss,
            'n_items': y[1:].nonzero(as_tuple=False).size(0),
        }
