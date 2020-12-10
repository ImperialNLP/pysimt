"""Character error rate (CER)."""

from typing import Iterable, Union, Optional
import editdistance

from .metric import Metric


class CERScorer:
    """Computes the character error rate (CER) metric and returns a `Metric`
    object.

    Args:
        refs: List of reference text files. Only the first one will be used
        hyps: Either a string denoting the hypotheses' filename, or
            a list that contains the hypotheses strings themselves
        language: unused
        lowercase: unused
    """
    def compute(self, refs: Iterable[str],
                hyps: Union[str, Iterable[str]],
                language: Optional[str] = None,
                lowercase: bool = False) -> Metric:
        if isinstance(hyps, str):
            # hyps is a file
            hyp_sents = open(hyps).read().strip().split('\n')
        elif isinstance(hyps, list):
            hyp_sents = hyps

        # refs is a list, take its first item
        with open(refs[0]) as f:
            ref_sents = f.read().strip().split('\n')

        assert len(hyp_sents) == len(ref_sents), "CER: # of sentences does not match."

        n_ref_chars = 0
        n_ref_tokens = 0
        dist_chars = 0
        dist_tokens = 0
        for hyp, ref in zip(hyp_sents, ref_sents):
            hyp_chars = hyp.split(' ')
            ref_chars = ref.split(' ')
            n_ref_chars += len(ref_chars)
            dist_chars += editdistance.eval(hyp_chars, ref_chars)

            # Convert char-based sentences to token-based ones
            hyp_tokens = hyp.replace(' ', '').replace('<s>', ' ').strip().split(' ')
            ref_tokens = ref.replace(' ', '').replace('<s>', ' ').strip().split(' ')
            n_ref_tokens += len(ref_tokens)
            dist_tokens += editdistance.eval(hyp_tokens, ref_tokens)

        cer = (100 * dist_chars) / n_ref_chars
        wer = (100 * dist_tokens) / n_ref_tokens

        verbose_score = "{:.3f}% (n_errors = {}, n_ref_chars = {}, WER = {:.3f}%)".format(
            cer, dist_chars, n_ref_chars, wer)

        return Metric('CER', cer, verbose_score, higher_better=False)
