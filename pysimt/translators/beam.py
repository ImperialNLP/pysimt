import logging
import math

import torch

from ..utils.device import DEVICE
from ..utils.io import progress_bar
from ..utils.data import sort_predictions
from ..models import SimultaneousTFNMT

logger = logging.getLogger('pysimt')


"""
Batched vanilla beam search without any simultaneous translation features.
"""
class BeamSearch:
    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 max_len=100, beam_size=5, lp_alpha=0., suppress_unk=False, n_best=False, **kwargs):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.filter_chain = filter_chain
        self.out_prefix = out_prefix
        self.beam_size = beam_size
        self.lp_alpha = lp_alpha
        self.suppress_unk = suppress_unk
        self.n_best = n_best

        self.vocab = self.model.trg_vocab
        self.n_vocab = len(self.vocab)
        self.unk = self.vocab['<unk>']
        self.eos = self.vocab['<eos>']
        self.bos = self.vocab['<bos>']
        self.pad = self.vocab['<pad>']

        self.max_len = max_len
        self.do_dump = out_prefix != ''
        
        self.is_transformer_model = isinstance(self.model, SimultaneousTFNMT)

    def dump_results(self, hyps, suffix=''):
        suffix = 'beam' if not suffix else f'{suffix}.beam'
        suffix += str(self.beam_size)

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

        return logp, h

    def decoder_init(self, state_dict=None):
        return self.model.dec.f_init(state_dict)

    def run_all(self):
        return self.run()

    def run(self, **kwargs):
        def tile_ctx_dict(ctx_dict, idxs):
            """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
            # 1st: tensor, 2nd optional mask

            if self.is_transformer_model:
                # TF
                return {
                    k: (t[:, idxs], None if mask is None else mask[idxs, :])
                    for k, (t, mask) in ctx_dict.items()
                }
            else:
                # RNN
                return {
                    k: (t[:, idxs], None if mask is None else mask[:, idxs])
                    for k, (t, mask) in ctx_dict.items()
                }

        def check_context_ndims(ctx_dict):
            for name, (ctx, mask) in ctx_dict.items():
                assert ctx.dim() == 3, \
                    f"{name}'s 1st dim should always be a time dimension."

        # effective batch size may be different
        max_batch_size = self.data_loader.batch_sampler.batch_size
        k = self.beam_size
        inf = -1000
        results = []

        # Common parts
        unk = self.unk
        eos = self.eos
        n_vocab = self.n_vocab

        # Tensorized beam that will shrink and grow up to max_batch_size
        beam_storage = torch.zeros(
            self.max_len, max_batch_size, k, dtype=torch.long, device=DEVICE)
        mask = torch.arange(max_batch_size * k, device=DEVICE)
        nll_storage = torch.zeros(max_batch_size, device=DEVICE)

        
        hyps_gs = torch.zeros(
            (self.max_len, max_batch_size), dtype=torch.long, device=DEVICE)

        for batch in progress_bar(self.data_loader, unit='batch'):
            batch.device(DEVICE)

            hyps_gs.zero_()

            # Always use the initial storage
            beam = beam_storage.narrow(1, 0, batch.size).zero_()

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = mask.narrow(0, 0, batch.size * k)

            # nll: batch_size x 1 (will get expanded further)
            nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

            # Tile indices to use in the loop to expand first dim
            tile = range(batch.size)

            # Cache encoder states
            self.model.cache_enc_states(batch)

            # Get encoder hidden states
            state_dict = self.model.get_enc_state_dict()

            # Sanity check one of the context dictionaries for dimensions
            check_context_ndims(state_dict)

            # Initial state is None i.e. 0. state_dict is not used
            h = self.decoder_init(state_dict)

            # last batch could be smaller than the requested batch size
            cur_batch_size = batch.size

            # Track sentences who already produced </s>
            track_fini = torch.zeros((cur_batch_size, k), device=DEVICE).bool()

            # Start all sentences with <s>
            next_word_idxs = self.model.get_bos(cur_batch_size).to(DEVICE)

            # The Transformer decoder require the <bos> to be passed alongside all hypothesis objects for prediction
            tf_decoder_input = next_word_idxs.unsqueeze(0)

            for t in range(self.max_len):
                if track_fini.all():
                    # All hypotheses produced </s>, early stop!
                    break


                state_dict = tile_ctx_dict(state_dict, tile)

                # logp: [B x V] or [B*k x V]
                # h is not used here in Transformer
                if self.is_transformer_model or t == 0:
                    # decoder initial hidden state may return NoneType
                    logp, h = self.decoder_step(
                        state_dict, next_word_idxs, h, tf_decoder_input)
                else:
                    logp, h = self.decoder_step(
                        state_dict, next_word_idxs, h[tile], tf_decoder_input)
                
                if self.suppress_unk:
                    logp[:, unk] = inf

                # Expand to 3D, cross-sum scores and reduce back to 2D
                # log_p: batch_size x vocab_size ( t = 0 )
                #   nll: batch_size x beam_size (x 1)
                # nll becomes: batch_size x beam_size*vocab_size here
                # Reduce (N, K*V) to k-best
                nll, beam[t] = nll.unsqueeze_(2).add(logp.view(
                    batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                        k, sorted=False, largest=True)
                    
                # previous indices into the beam and current token indices
                # TODO pay attention here, pytorch 1.4 compatible, need to test pytorch 1.8
                pdxs = torch.floor_divide(beam[t], n_vocab)
                beam[t].remainder_(n_vocab)
                next_word_idxs = beam[t].view(-1)

                # Update finished sentence tracker
                track_fini.add_(beam[t].eq(self.eos))

                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + (nk_mask // k) * (k if t else 1)
                
                tf_decoder_input = tf_decoder_input[:, tile]

                # Add the predicted word to the decoder's input. Used for the transformer models.
                tf_decoder_input = torch.cat((tf_decoder_input, next_word_idxs.unsqueeze(0)), dim=0)

                if t > 0:
                    # Permute all hypothesis history according to new order
                    beam[:t] = beam[:t].gather(2, pdxs.repeat(t, 1, 1))

                

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[self.max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

            if self.lp_alpha > 0.:
                len_penalty = ((5 + len_penalty)**self.lp_alpha) / 6**self.lp_alpha

            # Apply length normalization
            nll.div_(len_penalty)

            if self.n_best:
                # each elem is sample, then candidate
                tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
                scores = nll.to('cpu').tolist()
                results.extend(
                    [(self.vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
            else:
                # Get best-1 hypotheses
                top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
                hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
                results.extend(self.vocab.list_of_idxs_to_sents(hyps.tolist()))

        hyps = sort_predictions(self.data_loader, results)
        if self.do_dump:
            self.dump_results(hyps)

        return (hyps,)
