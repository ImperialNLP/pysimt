import time
import logging
import itertools

import torch

from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions

from .greedy import GreedySearch

logger = logging.getLogger('pysimt')


class SimultaneousGreedySearch(GreedySearch):
    ACT_READ, ACT_WRITE = 0, 1

    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 max_len=100, **kwargs):
        assert not model.opts.model['enc_bidirectional'], \
            "Bidirectional models can not be used for simultaneous MT."

        assert model.opts.model.get('dec_init', 'zero') == 'zero', \
            "`dec_init` should be 'zero' for simplicity."

        logger.info(f'Ignoring batch_size {batch_size} for simultaneous greedy search')
        batch_size = 1

        super().__init__(model, data_loader, out_prefix,
                         batch_size, filter_chain, max_len)

        # Partial modality i.e. text
        self._partial_key = str(model.sl)
        self.buffer = None

        self.list_of_s_0 = kwargs.pop('s_0', '').split(',')
        self.list_of_delta = kwargs.pop('delta', '').split(',')
        self.criteria = kwargs.pop('criteria', '').split(',')
        self.tf_decoder_input = None

    @staticmethod
    def wait_if_diff(cur_log_p, cur_next_pred, cand_log_p, cand_next_pred):
        """If the candidate changes with more context, READ. Otherwise WRITE."""
        return cand_next_pred.ne(cur_next_pred)

    @staticmethod
    def wait_if_worse(cur_log_p, cur_next_pred, cand_log_p, cand_next_pred):
        """If confidence for the candidate decreases WAIT/READ. Otherwise WRITE."""
        return cand_log_p[0, cur_next_pred] < cur_log_p[0, cur_next_pred]

    def write(self, new_word, new_h):
        """Write the new word, move the pointer and accept the hidden state."""
        self.prev_word, self.buffer[self.t_ptr] = new_word, new_word
        self.prev_h = new_h
        self.actions.append(self.ACT_WRITE)
        self.t_ptr += 1
        self.eos_written = new_word.item() == self.eos
        self.tf_decoder_input = torch.cat((self.tf_decoder_input, new_word.unsqueeze(0)), dim=0)

    def update_s(self, increment):
        """Update read pointer."""
        new_pos = min(self.s_len, self.s_ptr + increment)
        n_reads = new_pos - self.s_ptr
        self.actions.extend([self.ACT_READ] * n_reads)
        self.s_ptr = new_pos

    def clear_states(self):
        self.s_ptr = 0
        self.t_ptr = 0
        self.prev_h = None
        self._c_states = None
        self.prev_word = None
        self.eos_written = False
        self.actions = []

        if self.buffer is None:
            # Write buffer
            self.buffer = torch.zeros((self.max_len, ), dtype=torch.long, device=DEVICE)
        else:
            # Reset hypothesis buffer
            self.buffer.zero_()

    def is_src_read(self):
        return self.s_ptr >= self.s_len

    def cache_encoder_states(self, batch):
        """Encode full source sentence and cache the states."""
        self.model.cache_enc_states(batch)
        self.s_len = batch[self._partial_key].size(0)

    def read_more(self, n):
        """Reads more source words and computes new states."""
        return self.model.get_enc_state_dict(up_to=self.s_ptr + n)

    def get_src_prefix_str(self, batch):
        idxs = batch[self._partial_key][:self.s_ptr, 0].tolist()
        return self.model.vocabs[self._partial_key].idxs_to_sent(idxs)

    def get_trg_prefix_str(self):
        idxs = self.buffer[:self.t_ptr].tolist()
        return self.model.vocabs['trg'].idxs_to_sent(idxs)

    def run_all(self):
        """Do a grid search over the given list of parameters."""
        #############
        # grid search
        #############
        settings = itertools.product(
            self.list_of_s_0,
            self.list_of_delta,
            self.criteria,
        )
        for s_0, delta, crit in settings:
            # Run the decoding
            hyps, actions, up_time = self.run(int(s_0), int(delta), crit)

            # Dumps two files one with segmentations preserved, another
            # with post-processing filters applied
            self.dump_results(hyps, suffix=f's{s_0}_d{delta}_{crit}')

            # Dump actions
            self.dump_lines(actions, suffix=f's{s_0}_d{delta}_{crit}.acts')

    def run(self, s_0, delta, criterion):
        # R/W actions generated for the whole test set
        actions = []

        # Final translations
        translations = []

        # Set criterion
        crit_fn = getattr(self, criterion)

        start = time.time()
        for batch in pbar(self.data_loader, unit='batch'):
            self.clear_states()

            batch.device(DEVICE)

            # Compute all at once
            self.cache_encoder_states(batch)

            # Read some words and get trimmed states
            state_dict = self.read_more(s_0)
            self.update_s(s_0)

            # Initial state is None i.e. 0. state_dict is not used
            self.prev_h = self.decoder_init(state_dict)

            # last batch could be smaller than the requested batch size
            cur_batch_size = batch.size

            # Start all sentences with <s>
            self.set_first_word_to_bos(cur_batch_size)

            while not self.eos_written and self.t_ptr < self.max_len:
                logp, new_h, new_word = self.decoder_step(
                    state_dict, self.prev_word, self.prev_h, self.tf_decoder_input)
                if self.is_src_read():
                    # All source words are read, no choice but writing
                    self.write(new_word, new_h)
                else:
                    # C' is empty
                    if self._c_states is None:
                        self._c_states = self.read_more(delta)

                    # Evaluate candidate
                    cand_logp, cand_h, cand_new_word = self.decoder_step(
                        self._c_states, self.prev_word, self.prev_h, self.tf_decoder_input)

                    if crit_fn(logp, new_word, cand_logp, cand_new_word):
                        # Wait/Read more words and do another decoding attempt
                        state_dict = self._c_states
                        self._c_states = None
                        self.update_s(delta)
                    else:
                        # Commit the last candidate
                        self.write(new_word, new_h)

            # All finished, convert translations to python lists on CPU
            idxs = self.buffer[self.buffer.ne(0)].tolist()
            if idxs[-1] != self.eos:
                # In cases where <eos> not produced and the above loop
                # went on until max_len, add an explicit <eos> for correctness
                idxs.append(self.eos)

            # compute action sequence from which metrics will be computed
            actions.append(' '.join(map(lambda i: str(i), self.actions)))
            translations.append(self.vocab.idxs_to_sent(idxs))

        up_time = time.time() - start

        hyps = sort_predictions(self.data_loader, translations)
        actions = sort_predictions(self.data_loader, actions)
        return (hyps, actions, up_time)

    def set_first_word_to_bos(self, cur_batch_size):
        self.prev_word = self.model.get_bos(cur_batch_size).to(DEVICE)
        self.tf_decoder_input = self.prev_word.unsqueeze(0)
