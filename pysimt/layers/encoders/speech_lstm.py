from typing import Optional

from torch import nn
from torch.nn import functional as F

from ..ff import FF


class SpeechLSTM(nn.Module):
    """A bidirectional LSTM encoder with subsampling for speech features.

    The number of LSTM layers is defined by the `layers` argument, i.e.
    `1_1_2_2_1_1` denotes 6 LSTM layers where the middle two applies
    a subsampling factor of 2 to their inputs. Subsampling in this context
    means that every N'th state will be passed to the next layer as input.

    Each LSTM layer is followed by a feed-forward projection layer whose
    non-linearity is given by the `activ` argument.

    Note:
        The input tensor should contain samples of equal lengths i.e.
        `bucket_by` in training configuration should be set to the acoustic
        features modality.

    Args:
        input_size: Input feature dimensionality.
        hidden_size: LSTM hidden state dimensionality.
        proj_size: Projection layer size.
        activ: Non-linearity to apply to intermediate projection
            layers. (Default: 'tanh')
        layers: A '_' separated list of integers that defines the subsampling
            factor for each LSTM.
        dropout: Use dropout (Default: 0.)

    Input:
        x: A `torch.Tensor` of shape `(n_timesteps, n_samples, input_size)`

    Output:
        hs: A `torch.Tensor` of shape `(n_timesteps, n_samples, hidden_size * 2)`
            that contains encoder hidden states for all timesteps.
        mask: `None` since this layer expects all equal frame inputs.

    """
    def __init__(self, input_size: int, hidden_size: int, proj_size: int,
                 layers: str, activ: Optional[str] = 'tanh',
                 dropout: float = 0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.activ = activ
        self.layers = [int(i) for i in layers.split('_')]
        self.dropout = dropout
        self.n_layers = len(self.layers)

        # Doubles its size because of concatenation of forw-backw encs
        self.ctx_size = self.hidden_size * 2

        # Fill 0-vector as <eos> to the end of the frames
        self.pad_tuple = (0, 0, 0, 0, 0, 1)

        # Projections and LSTMs
        self.ffs = nn.ModuleList()
        self.lstms = nn.ModuleList()

        if self.dropout > 0:
            self.do = nn.Dropout(self.dropout)

        for i, ss_factor in enumerate(self.layers):
            # Add LSTMs
            self.lstms.append(nn.LSTM(
                self.input_size if i == 0 else self.hidden_size,
                self.hidden_size, bidirectional=True))
            # Add non-linear bottlenecks
            self.ffs.append(FF(
                self.ctx_size, self.proj_size, activ=self.activ))

    def forward(self, x, **kwargs):
        # Generate a mask to detect padded sequences
        mask = x.ne(0).float().sum(2).ne(0).float()

        if mask.eq(0).nonzero().numel() > 0:
            raise RuntimeError("Non-homogeneous batch detected in SpeechLSTM layer.")

        # Pad with <eos> zero
        hs = F.pad(x, self.pad_tuple)

        for (ss_factor, f_lstm, f_ff) in zip(self.layers, self.lstms, self.ffs):
            if ss_factor > 1:
                # Skip states
                hs = f_ff(f_lstm(hs[::ss_factor])[0])
            else:
                hs = f_ff(f_lstm(hs)[0])

        if self.dropout > 0:
            hs = self.do(hs)

        # No mask is returned as batch should contain same-length sequences
        return hs, None
