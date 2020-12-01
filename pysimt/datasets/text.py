import logging
from pathlib import Path

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..utils.data import read_sentences

logger = logging.getLogger('pysimt')


class TextDataset(Dataset):
    r"""A PyTorch dataset for sentences.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object giving
            the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, a special beginning-of-sentence
            "<bos>" marker will be prepended to sentences.
    """

    def __init__(self, fname, vocab, bos=False, eos=True, **kwargs):
        self.path = Path(fname)
        self.vocab = vocab
        self.bos = bos
        self.eos = eos

        # Detect glob patterns
        self.fnames = sorted(self.path.parent.glob(self.path.name))

        if len(self.fnames) == 0:
            raise RuntimeError('{} does not exist.'.format(self.path))
        elif len(self.fnames) > 1:
            logger.info('Multiple files found, using first: {}'.format(self.fnames[0]))

        # Read the sentences and map them to vocabulary
        self.data, self.lengths = read_sentences(
            self.fnames[0], self.vocab, bos=self.bos, eos=self.eos)

        # Dataset size
        self.size = len(self.data)

    @staticmethod
    def to_torch(batch, **kwargs):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)".format(
            self.__class__.__name__, self.fnames[0].name, self.__len__())
        return s
