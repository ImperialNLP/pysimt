import logging
from pathlib import Path
from typing import Tuple, List

import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..utils.io import fopen
from ..vocabulary import Vocabulary

logger = logging.getLogger('pysimt')


class TextDataset(Dataset):
    """A convenience dataset for reading monolingual text files.

    Args:
        fname: A string or ``pathlib.Path`` object giving
            the corpus.
        vocab: A ``Vocabulary`` instance for the given corpus.
        bos: Optional; If ``True``, a special beginning-of-sentence
            `<bos>` marker will be prepended to sequences.
        eos: Optional; If ``True``, a special end-of-sentence
            `<eos>` marker will be appended to sequences.
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
        self.data, self.lengths = self.read_sentences(
            self.fnames[0], self.vocab, bos=self.bos, eos=self.eos)

        # Dataset size
        self.size = len(self.data)

    @staticmethod
    def read_sentences(fname: str,
                       vocab: Vocabulary,
                       bos: bool = False,
                       eos: bool = True) -> Tuple[List[List[int]], List[int]]:
        lines = []
        lens = []
        with fopen(fname) as f:
            for idx, line in enumerate(progress_bar(f, unit='sents')):
                line = line.strip()

                # Empty lines will cause a lot of headaches,
                # get rid of them during preprocessing!
                assert line, "Empty line (%d) found in %s" % (idx + 1, fname)

                # Map and append
                seq = vocab.sent_to_idxs(line, explicit_bos=bos, explicit_eos=eos)
                lines.append(seq)
                lens.append(len(seq))

        return lines, lens


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
