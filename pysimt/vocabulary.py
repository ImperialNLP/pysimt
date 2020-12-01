import json
import pathlib
import logging
from collections import OrderedDict
from typing import List

logger = logging.getLogger('pysimt')


class Vocabulary:
    r"""Vocabulary class for integer<->token mapping.

    Arguments:
        fname (str): The filename of the JSON vocabulary file created by
            `pysimt-build-vocab` script.
        short_list (int, optional): If > 0, only the most frequent `short_list`
            items are kept in the vocabulary.

    Attributes:
        vocab (pathlib.Path): A :class:`pathlib.Path` instance holding the
            filepath of the vocabulary file.
        short_list (int): Short-list threshold.
        freqs (dict): A dictionary which maps vocabulary strings to their
            normalized frequency across the training set.
        counts (dict): A dictionary which maps vocabulary strings to their
            occurrence counts across the training set.
        n_tokens (int): The final number of elements in the vocabulary.
        has_bos (bool): `True` if the vocabulary has <bos> token.
        has_eos (bool): `True` if the vocabulary has <eos> token.
        has_pad (bool): `True` if the vocabulary has <pad> token.
        has_unk (bool): `True` if the vocabulary has <unk> token.

    Note:
        The final instance can be easily queried in both directions with
        bracket notation using integers and strings.

    Example:
        >>> vocab = Vocabulary('train.vocab.en')
        >>> vocab['woman']
        23
        >>> vocab[23]
        'woman'

    Returns:
        A :class:`Vocabulary` instance.
    """

    TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self, fname: str, short_list: int = 0):
        self.vocab = pathlib.Path(fname).expanduser()
        self.short_list = short_list
        self._map = None
        self._imap = None
        self.freqs = None
        self.counts = None
        self._allmap = None
        self.n_tokens = None

        self.has_pad = False
        self.has_bos = False
        self.has_eos = False
        self.has_unk = False

        # Load file
        with open(self.vocab) as f:
            data = json.load(f)

        if self.short_list > 0:
            # Get a slice of most frequent `short_list` items
            data = dict(list(data.items())[:self.short_list])

        self._map = {k: int(v.split()[0]) for k, v in data.items()}
        self.counts = {k: int(v.split()[1]) for k, v in data.items()}

        total_count = sum(self.counts.values())
        self.freqs = {k: v / total_count for k, v in self.counts.items()}

        # Sanity check for placeholder tokens
        for tok, idx in self.TOKENS.items():
            if self._map.get(tok, -1) != idx:
                logger.info(f'{tok} not found in {self.vocab.name!r}')
                setattr(self, f'has_{tok[1:-1]}', False)
            else:
                setattr(self, f'has_{tok[1:-1]}', True)

        # Set # of tokens
        self.n_tokens = len(self._map)

        # Invert dictionary
        self._imap = OrderedDict([(v, k) for k, v in self._map.items()])

        # Merge forward and backward lookups into single dict for convenience
        self._allmap = OrderedDict()
        self._allmap.update(self._map)
        self._allmap.update(self._imap)

        assert len(self._allmap) == (len(self._map) + len(self._imap)), \
            "Merged vocabulary size is not equal to sum of both."

    def sent_to_idxs(self, line: str, explicit_bos: bool = False,
                     explicit_eos: bool = True) -> List[int]:
        """Convert from list of strings to list of token indices."""
        tidxs = []

        if explicit_bos and self.has_bos:
            tidxs.append(self.TOKENS["<bos>"])

        if self.has_unk:
            for tok in line.split():
                tidxs.append(self._map.get(tok, self.TOKENS["<unk>"]))
        else:
            # Remove unknown tokens from the words
            for tok in line.split():
                try:
                    tidxs.append(self._map[tok])
                except KeyError:
                    # make this verbose and repetitive as this should be
                    # used cautiously only for some specific models
                    logger.info('No <unk> token, removing word from sentence')

        if explicit_eos and self.has_eos:
            tidxs.append(self.TOKENS["<eos>"])

        return tidxs

    def idxs_to_sent(self, idxs: List[int], debug: bool = False) -> str:
        r"""Converts list of integers to string representation.

        Arguments:
            idxs (List[int]): Python list of integers as previously mapped from
                string tokens by this instance.
            debug (bool, optional): If `True`, the string representation
                will go beyond and include the end-of-sentence token as well.

        Returns:
            A whitespace separated string representing the given list of integers.

        """
        result = []
        for idx in idxs:
            if not debug and self.has_eos and idx == self.TOKENS["<eos>"]:
                break
            result.append(self._imap.get(idx, self.TOKENS["<unk>"]))

        return " ".join(result)

    def list_of_idxs_to_sents(self, lidxs: List[List[int]]) -> List[str]:
        r"""Converts list of list of integers to string representations. This is
        handy for batched conversion after beam search for example.

        Arguments:
            lidxs(List[List[int]]): A list containing multiple lists of integers as
                previously mapped from string tokens by this instance.

        Returns:
            A list of whitespace separated strings representing the given input.

        """
        results = []
        unk = self.TOKENS["<unk>"]
        for idxs in lidxs:
            result = []
            for idx in idxs:
                if idx == self.TOKENS["<eos>"]:
                    break
                result.append(self._imap.get(idx, unk))
            results.append(" ".join(result))
        return results

    def __getitem__(self, key):
        return self._allmap[key]

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        return f"Vocabulary of {self.n_tokens} items ({self.vocab.name!r})"
