from sacrebleu import corpus_bleu

from ..utils.misc import listify
from ..utils.io import read_reference_files, read_hypothesis_file
from .metric import Metric


class BLEUScorer:
    """Computes the multi-bleu equivalent using SacreBLEU, with tokenization
    option disabled."""
    def compute(self, refs, hyps, language=None):
        if isinstance(hyps, str):
            hyps = read_hypothesis_file(hyps)

        assert isinstance(hyps, list)

        refs = read_reference_files(*listify(refs))

        score = corpus_bleu(hyps, refs, tokenize='none')
        verbose_score = ' '.join(score.format().split()[2:])
        float_score = score.score
        return Metric('BLEU', float_score, verbose_score)
