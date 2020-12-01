#!/usr/bin/env python
import os
import sys

from pathlib import Path

import tabulate
import sacrebleu

from pysimt.metrics.simnmt import AVPScorer, AVLScorer


"""This script should be run from within the parent folder where each pysimt
experiment resides."""


def read_lines_from_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines


if __name__ == '__main__':
    results = {}
    trglang = sys.argv[1]
    if trglang not in ('en', 'de', 'fr', 'cs'):
        print(f'Usage: {sys.argv[0]} <target lang> [action files]')
        sys.exit(1)

    scorers = [
        AVPScorer(add_trg_eos=False),
        AVLScorer(add_trg_eos=False),
    ]

    act_files = sys.argv[2:]

    # get test set
    test_sets = set([a.split('.')[1] for a in act_files])
    assert len(test_sets) == 1, "Different test set files given"
    test_set = list(test_sets)[0]
    print(f'Test set is {test_set}, target language is {trglang}\n\n')

    ref_root = Path(__file__).parent / f'../data/multi30k/en-{trglang}'
    ref_file = ref_root / f'{test_set}.lc.norm.tok.{trglang}.dehyph'
    if ref_file.exists():
        refs = read_lines_from_file(ref_file)
    else:
        raise RuntimeError(f'{ref_file} does not exist')

    for act_file in act_files:
        # Compute delay metrics
        scores = [s.compute_from_file(act_file) for s in scorers]
        results[act_file] = {s.name: s.score for s in scores}

        # try to reach hypothesis file
        hyp_file = act_file.replace('.acts', '.gs')
        if os.path.exists(hyp_file):
            hyps = read_lines_from_file(hyp_file)
            bleu = sacrebleu.corpus_bleu(
                hyps, [refs], tokenize='none', lowercase=False).score
        else:
            bleu = -1.0

        results[act_file]['BLEU'] = bleu
        results[act_file]['Q/AVP'] = bleu / scores[0].score

    if results:
        headers = ['Name'] + list(next(iter(results.values())).keys())
        results = [[name, *[scores[key] for key in headers[1:]]] for name, scores in results.items()]
        results = sorted(results, key=lambda x: x[headers.index('BLEU')])
        print(tabulate.tabulate(results, headers=headers, floatfmt='.2f'))
