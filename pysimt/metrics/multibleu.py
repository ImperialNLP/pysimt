"""Tokenized BLEU through sacreBLEU API."""

from typing import Union, Iterable, TextIO

from sacrebleu import corpus_bleu

from ..utils.misc import listify
from ..utils.io import read_reference_files, read_hypothesis_file
from .metric import Metric


class BLEUScorer:
    """Computes the multi-bleu equivalent using SacreBLEU, with tokenization
    option disabled.

    Args:
        refs: List of reference text files
        hyps: A file path, or a list of hypothesis strings or an open file handle
        language: unused
    """
    def compute(self, refs: Iterable[str],
                hyps: Union[str, Iterable[str], TextIO],
                language=None) -> Metric:
        if isinstance(hyps, str):
            hyps = read_hypothesis_file(hyps)

        assert isinstance(hyps, list)

        refs = read_reference_files(*listify(refs))

        score = corpus_bleu(hyps, refs, tokenize='none')
        verbose_score = ' '.join(score.format().split()[2:])
        float_score = score.score
        return Metric('BLEU', float_score, verbose_score)
