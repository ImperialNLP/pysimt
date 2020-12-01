import logging
import math

import torch

from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions

logger = logging.getLogger('pysimt')


"""Batched vanilla greedy search without any simultaneous translation
features."""


class GreedySearch:
    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 max_len=100, **kwargs):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.filter_chain = filter_chain
        self.out_prefix = out_prefix

        self.vocab = self.model.trg_vocab
        self.n_vocab = len(self.vocab)
        self.unk = self.vocab['<unk>']
        self.eos = self.vocab['<eos>']
        self.bos = self.vocab['<bos>']
        self.pad = self.vocab['<pad>']

        self.max_len = max_len
        self.do_dump = out_prefix != ''

    def dump_results(self, hyps, suffix=''):
        suffix = 'gs' if not suffix else f'{suffix}.gs'

        # Dump raw ones (BPE/SPM etc.)
        self.dump_lines(hyps, suffix + '.raw')
        if self.filter_chain is not None:
            self.dump_lines(self.filter_chain.apply(hyps), suffix)

    def dump_lines(self, lines, suffix):
        fname = f'{self.out_prefix}.{suffix}'
        with open(fname, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')

    def decoder_step(self, state_dict, next_word_idxs, h, hypothesis=None):
        logp, h = self.model.dec.f_next(
            state_dict, self.model.dec.get_emb(next_word_idxs), h, hypothesis)

        # Similar to the logic in fairseq https://bit.ly/3agXAa7
        # Never select the pad token or the bos token
        logp[:, self.pad] = -math.inf
        logp[:, self.bos] = -math.inf

        # Compute most likely word idxs
        next_word_idxs = logp.argmax(dim=-1)
        return logp, h, next_word_idxs

    def decoder_init(self, state_dict=None):
        return self.model.dec.f_init(state_dict)

    def run_all(self):
        return self.run()

    def run(self, **kwargs):
        # effective batch size may be different
        max_batch_size = self.data_loader.batch_sampler.batch_size

        translations = []
        hyps = torch.zeros(
            (self.max_len, max_batch_size), dtype=torch.long, device=DEVICE)

        for batch in pbar(self.data_loader, unit='batch'):
            batch.device(DEVICE)

            # Reset hypotheses
            hyps.zero_()

            # Cache encoder states
            self.model.cache_enc_states(batch)

            # Get encoder hidden states
            state_dict = self.model.get_enc_state_dict()

            # Initial state is None i.e. 0. state_dict is not used
            h = self.decoder_init(state_dict)

            # last batch could be smaller than the requested batch size
            cur_batch_size = batch.size

            # Track sentences who already produced </s>
            track_fini = torch.zeros((cur_batch_size, ), device=DEVICE).bool()

            # Start all sentences with <s>
            next_word_idxs = self.model.get_bos(cur_batch_size).to(DEVICE)

            # The Transformer decoder require the <bos> to be passed alongside all hypothesis objects for prediction
            tf_decoder_input = next_word_idxs.unsqueeze(0)

            # A maximum of `max_len` decoding steps
            for t in range(self.max_len):
                if track_fini.all():
                    # All hypotheses produced </s>, early stop!
                    break

                logp, h, next_word_idxs = self.decoder_step(
                    state_dict, next_word_idxs, h, tf_decoder_input)

                # Update finished sentence tracker
                track_fini.add_(next_word_idxs.eq(self.eos))

                # Insert most probable words for timestep `t` into tensor
                hyps[t, :cur_batch_size] = next_word_idxs

                # Add the predicted word to the decoder's input. Used for the transformer models.
                tf_decoder_input = torch.cat((tf_decoder_input, next_word_idxs.unsqueeze(0)), dim=0)

            # All finished, convert translations to python lists on CPU
            sent_idxs = hyps[:, :cur_batch_size].t().cpu().tolist()
            translations.extend(self.vocab.list_of_idxs_to_sents(sent_idxs))

        hyps = sort_predictions(self.data_loader, translations)

        if self.do_dump:
            self.dump_results(hyps)

        return (hyps,)
