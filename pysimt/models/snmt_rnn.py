# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import RecurrentEncoder, VisualFeaturesEncoder
from ..layers.decoders import ConditionalGRUDecoder
from ..vocabulary import Vocabulary
from ..utils.nn import get_n_params
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.io import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('pysimt')

"""You can use this model to pre-train a unidirectional NMT model. Once trained,
you can decode translations from this model either using plain greedy-search
or state-of-the-art simultanous decoding algorithms.

# Pure greedy-search (by default batched)
$ pysimt stranslate -s test_2016_flickr,test_2017_flickr \
    -f gs -o <output_prefix> <best model's .ckpt>

"""


class SimultaneousNMT(nn.Module):
    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'enc_dim': 256,             # Encoder hidden size
            'enc_proj_dim': None,       # Encoder final projection
            'enc_proj_activ': 'linear',  # Encoder final projection activation
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'enc_bidirectional': True,  # Whether the RNN encoder should be bidirectional
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',      # The only option is `cond`
            'dec_inp_activ': None,
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'short_list': 0,            # Short list vocabularies (0: disabled)
            'out_logic': 'simple',      # 'simple' or 'deep' output
            # Visual features (optional)
            'aux_dim': None,            # Auxiliary features dim (# channels for conv features)
            'aux_dropout': 0.0,         # Auxiliary features dropout
            'aux_lnorm': False,         # layer-norm
            'aux_l2norm': False,        # L2-normalize
            'aux_proj_dim': None,       # Projection layer for features
            'aux_proj_activ': None,     # Projection layer non-linearity
            'num_regions': 36,          # The number of regions to use. Valid only for OD features. Default: 36.
            'feat_mode': None,          # OD feature type. None defaults to `roi_feats`
            'mm_fusion_op': 'concat',   # fusion type
            'mm_fusion_dropout': 0.0,   # fusion dropout
            # Decoding/training simultaneous NMT args
            'translator_type': 'gs',    # This model implements plain unidirectional MT
                                        # so the decoding is normal greedy-search
            'translator_args': {},      # No extra arguments to translator
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, short_list=self.opts.model['short_list'])

        # Inherently non multi-lingual aware
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            self.tl = tlangs[0]
            self.trg_vocab = self.vocabs[self.tl]
            self.n_trg_vocab = len(self.trg_vocab)
            self.val_refs = self.opts.data['val_set'][self.tl]

        # Check vocabulary sizes for 3way tying
        if self.opts.model.get('tied_emb', False) not in [False, '2way', '3way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        if self.opts.model.get('tied_emb', False) == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        for layer in list(self.encoders.values()) + [self.dec]:
            if hasattr(layer, 'emb'):
                with torch.no_grad():
                    layer.emb.weight.data[0].fill_(0)

    def create_src_encoder(self):
        """Creates and returns an RNN encoder for textual input."""
        return RecurrentEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            proj_dim=self.opts.model['enc_proj_dim'],
            proj_activ=self.opts.model['enc_proj_activ'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            layer_norm=self.opts.model['enc_lnorm'],
        )

    def create_image_encoder(self):
        """Creates and returns an MLP encoder for visual features."""
        return VisualFeaturesEncoder(
            input_size=self.opts.model['aux_dim'],
            proj_dim=self.opts.model['aux_proj_dim'],
            proj_activ=self.opts.model['aux_proj_activ'],
            layer_norm=self.opts.model['aux_lnorm'],
            l2_norm=self.opts.model['aux_l2norm'],
            dropout=self.opts.model['aux_dropout'],
        )

    def create_decoder(self, encoders):
        """Creates and returns the RNN decoder. No hidden state initialization
        for sake of simplicity. Encoders are passed to allow multi-modal
        attention out-of-the-box."""
        return ConditionalGRUDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            encoders=encoders,
            rnn_type=self.opts.model['dec_type'],
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'],
            mm_fusion_op=self.opts.model['mm_fusion_op'],
            mm_fusion_dropout=self.opts.model['mm_fusion_dropout'],
        )

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        encoders = {}
        for key in self.topology.srcs.keys():
            encoders[key] = getattr(self, f'create_{key}_encoder')()
        self.encoders = nn.ModuleDict(encoders)
        self.dec = self.create_decoder(encoders=self.encoders)

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.encoders[str(self.sl)].emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        # For wait_if_diff, wait_if_worse and test-time waitk decodings
        if mode == 'beam' and self.opts.model['translator_type'] != 'gs':
            batch_size = 1

        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            max_len=self.opts.model['max_len'],
            sampler_type=self.opts.model['sampler_type'],
            bucket_by=self.opts.model['bucket_by'],
            bucket_order=self.opts.model['bucket_order'],
            # order_file is for multimodal adv. evaluation
            order_file=self.opts.data[split + '_set'].get('ord', None),
            feat_mode=self.opts.model['feat_mode'],
            num_regions=self.opts.model['num_regions'])
        logger.info(self.dataset)
        return self.dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def cache_enc_states(self, batch):
        """Caches encoder states internally by forward-pass'ing each encoder."""
        for key, enc in self.encoders.items():
            _ = enc(batch[key])

    def get_enc_state_dict(self, up_to=int(1e6)):
        """Encodes the batch optionally by partial encoding up to `up_to`
        words for derived simultaneous NMT classes. By default, the value
        is large enough to leave it as vanilla NMT."""
        return {str(k): e.get_states(up_to=up_to) for k, e in self.encoders.items()}

    def forward(self, batch, **kwargs):
        """Training forward-pass with explicit timestep-based loop."""
        loss = 0.0

        # Cache encoder states first
        self.cache_enc_states(batch)

        # Encode modalities and get the dict back
        state_dict = self.get_enc_state_dict()

        # Initial state is None i.e. 0. `state_dict` is not used
        h = self.dec.f_init(state_dict)

        # Convert target token indices to embeddings -> T*B*E
        y = batch[self.tl]
        y_emb = self.dec.emb(y)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.size(0) - 1):
            log_p, h = self.dec.f_next(state_dict, y_emb[t], h)
            loss += self.dec.nll_loss(log_p, y[t + 1])

        return {
            'loss': loss,
            'n_items': y[1:].nonzero(as_tuple=False).size(0),
        }

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def register_tensorboard(self, handle):
        """Stores tensorboard hook for custom logging."""
        self.tboard = handle
