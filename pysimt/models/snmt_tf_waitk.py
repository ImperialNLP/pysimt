import logging

from . import SimultaneousTFNMT

logger = logging.getLogger('pysimt')

"""This is the training-time wait-k model from:
    Ma et al. (2018), STACL: Simultaneous Translation with Implicit Anticipation
   and Controllable Latency using Prefix-to-Prefix Framework, arXiv:1810.08398

The only required parameter is the `k` argument for training. When decoding,
pass the `k` argument explicitly to `pysimt stranslate`. A large enough `k`
should produce the same results as the `snmt.py` model.
"""


class SimultaneousTFWaitKNMT(SimultaneousTFNMT):

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            # Decoding/training simultaneous NMT args
            'translator_type': 'wk',  # This model implements train-time wait-k
            'translator_args': {'k': 1e4},  # k as in wait-k in training
            'consecutive_warmup': 0,  # consecutive training for this many epochs
        })

    def __init__(self, opts):
        super().__init__(opts)
        assert not self.opts.model['enc_bidirectional'], \
            'Bidirectional TF encoder is not currently supported for simultaneous MT.'

    def forward(self, batch, **kwargs):
        """
        Performs a forward pass.
        :param batch: The batch.
        :param kwargs: Any extra arguments.
        :return: The output from the forward pass.
        """
        k = int(self.opts.model['translator_args']['k'])
        if self.training:
            epoch_count = kwargs['ectr']
            if epoch_count <= self.opts.model['consecutive_warmup']:
                # warming up, use full contexts
                k = int(1e4)

        # Pass 'k' to the model.
        return super().forward(batch, k=k)
