import logging
import time

from .sim_greedy import SimultaneousGreedySearch
from ..utils.data import sort_predictions
from ..utils.device import DEVICE
from ..utils.io import pbar

logger = logging.getLogger('pysimt')


class SimultaneousWaitKGreedySearch(SimultaneousGreedySearch):
    def run_all(self):
        """Do a grid search over the given list of parameters."""

        # Let's pretend that `s_0` is `k`
        for k in self.list_of_s_0:
            # Run the decoding
            hyps, actions, up_time = self.run(int(k))

            # Dumps two files one with segmentations preserved, another
            # with post-processing filters applied
            self.dump_results(hyps, suffix=f'wait{k}')

            # Dump actions
            self.dump_lines(actions, suffix=f'wait{k}.acts')

    def run(self, k):
        # R/W actions generated for the whole test set
        actions = []

        # Final translations
        translations = []

        start = time.time()
        for batch in pbar(self.data_loader, unit='batch'):
            self.clear_states()

            batch.device(DEVICE)

            # Compute all at once
            self.cache_encoder_states(batch)

            # Read some words and get trimmed states
            state_dict = self.read_more(k)
            self.update_s(k)

            # Initial state is None i.e. 0. state_dict is not used
            self.prev_h = self.decoder_init(state_dict)

            # last batch could be smaller than the requested batch size
            cur_batch_size = batch.size

            # Start all sentences with <s>
            self.set_first_word_to_bos(cur_batch_size)

            # We will start by writing
            next_action = self.ACT_WRITE

            while not self.eos_written and self.t_ptr < self.max_len:
                if next_action == self.ACT_WRITE or self.is_src_read():
                    logp, new_h, new_word = self.decoder_step(
                        state_dict, self.prev_word, self.prev_h, self.tf_decoder_input)
                    self.write(new_word, new_h)
                else:
                    # READ
                    state_dict = self.read_more(1)
                    self.update_s(1)

                # Invert the last committed action for interleaved decoding
                next_action = 1 - self.actions[-1]

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
