from sacrebleu import corpus_bleu

from ..utils.misc import listify
from ..utils.io import read_reference_files, read_hypothesis_file
from .metric import Metric


class SACREBLEUScorer:
    """Computes the usual SacreBLEU metric with the default v13a tokenizer.
    This metric expects de-tokenized references and hypotheses, i.e.
    it is only sensible to use this with SPM files and the de-spm
    post-processing filter. For the more usual tokenized BLEU, check the
    BLEU metric.
    """
    def compute(self, refs, hyps, language=None):
        if isinstance(hyps, str):
            hyps = read_hypothesis_file(hyps)

        assert isinstance(hyps, list)

        refs = read_reference_files(*listify(refs))

        score = corpus_bleu(hyps, refs)
        verbose_score = ' '.join(score.format().split()[2:])
        float_score = score.score
        return Metric('SACREBLEU', float_score, verbose_score)
