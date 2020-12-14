import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

from ...utils.nn import get_activation_fn
from .. import FF, Fusion
from ..attention import get_attention


class ConditionalGRUDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial. It supports
    multimodal attention if more than one source modality is available. The
    initial state of the decoder RNN is set to `zero` and can not be modified
    for the sake of simplicity for simultaneous MT."""

    def __init__(self, input_size, hidden_size, n_vocab, encoders,
                 rnn_type='gru', tied_emb=False, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 dropout_out=0, out_logic='simple', dec_inp_activ=None,
                 mm_fusion_op=None, mm_fusion_dropout=0.0):
        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()

        # Safety checks
        assert self.rnn_type in ('GRU',), f"{rnn_type!r} unknown"
        assert mm_fusion_op in ('sum', 'concat', None), "mm_fusion_op unknown"

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab
        self.dec_inp_activ_fn = get_activation_fn(dec_inp_activ)

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size, padding_idx=0)

        # Create attention layer(s)
        self.att = nn.ModuleDict()

        for key, enc in encoders.items():
            Attention = get_attention(att_type)
            self.att[str(key)] = Attention(
                enc.ctx_size,
                self.hidden_size,
                transform_ctx=True,
                ctx2hid=True,
                mlp_bias=False,
                att_activ=att_activ,
                att_bottleneck=att_bottleneck,
                temp=att_temp)

        # return the only c_t from the list
        self.fusion = lambda x: x[0]
        if len(encoders) > 1:
            # Multiple inputs (multimodal NMT)
            ctx_sizes = [ll.ctx_size for ll in encoders.values()]
            if mm_fusion_op == 'concat':
                mm_inp_size = sum(ctx_sizes)
            else:
                assert len(set(ctx_sizes)) == 1, \
                    "Context sizes are not compatible with mm_fusion_op!"
                mm_inp_size = ctx_sizes[0]

            fusion = [Fusion(mm_fusion_op, input_size=mm_inp_size, output_size=self.hidden_size)]
            if mm_fusion_dropout > 0:
                fusion.append(nn.Dropout(mm_fusion_dropout))
            self.fusion = nn.Sequential(*fusion)

        # Create decoders
        self.dec0 = RNN(self.input_size, self.hidden_size)
        self.dec1 = RNN(self.hidden_size, self.hidden_size)

        # Output dropout
        if dropout_out > 0:
            self.do_out = nn.Dropout(p=dropout_out)
        else:
            self.do_out = lambda x: x

        # Output bottleneck: maps hidden states to target emb dim
        # simple: tanh(W*h)
        #   deep: tanh(W*h + U*emb + V*ctx)
        out_inp_size = self.hidden_size

        # Dummy op to return back the hidden state for simple output
        self.out_merge_fn = lambda h, e, c: h

        if out_logic == 'deep':
            out_inp_size += (self.input_size + self.hidden_size)
            self.out_merge_fn = lambda h, e, c: torch.cat((h, e, c), dim=1)

        # Final transformation that receives concatenated outputs or only h
        self.hid2out = FF(out_inp_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def get_emb(self, idxs):
        """Returns time-step based embeddings."""
        return self.emb(idxs)

    def f_init(self, state_dict=None):
        """Returns the initial h_0 for the decoder."""
        self.history = defaultdict(list)
        return None

    def f_next(self, state_dict, y, h, hypothesis=None):
        """Applies one timestep of recurrence. `state_dict` may contain
        partial source information depending on how the model constructs it."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1 = self.dec0(y, h)
        query = h1.unsqueeze(0)

        # Obtain attention for each different input context in encoder
        atts = []
        for k, (s, m) in state_dict.items():

            alpha, ctx = self.att[k](query, s, m)
            atts.append(ctx)

            if not self.training:
                self.history[f'alpha_{k}'].append(alpha.cpu())

        # Fuse input contexts
        c_t = self.fusion(atts)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        # Additional optional transformation is to make the comparison
        # fair with the MMT model.
        h2 = self.dec1(self.dec_inp_activ_fn(c_t), h1)

        # Output logic: dropout -> proj(o_t)
        # transform logit to T*B*V (V: vocab_size)
        logit = self.out2prob(
            self.do_out(self.hid2out(self.out_merge_fn(h2, y, c_t))))

        # Compute log_softmax over token dim
        log_p = F.log_softmax(logit, dim=-1)

        # Return log probs and new hidden states
        return log_p, h2
