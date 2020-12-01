#!/usr/bin/env python
import os
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from hashlib import sha1

import numpy as np

import sacrebleu
import tabulate

from pysimt.metrics.simnmt import AVPScorer, AVLScorer, CWMScorer, CWXScorer


"""This script should be run from within the parent folder where each pysimt
experiment resides."""


def read_lines_from_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines


def compute_bleu(fname, refs):
    hyps = open(fname).read()
    hashsum = sha1(hyps.encode('utf-8')).hexdigest()
    parent = fname.parent
    cached_bleu = parent / f'.{fname.name}__{hashsum}'
    if os.path.exists(cached_bleu):
        return float(open(cached_bleu).read().strip().split()[2])
    else:
        bleu = sacrebleu.corpus_bleu(
            hyps.strip().split('\n'), refs, tokenize='none')
        with open(cached_bleu, 'w') as f:
            f.write(bleu.format() + '\n')
        return float(bleu.format().split()[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='delay-analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compute delay metrics for multiple runs",
        argument_default=argparse.SUPPRESS)

    parser.add_argument('-r', '--ref-file', required=True, type=str,
                        help='The reference file for BLEU evaluation.')
    parser.add_argument('act_files', nargs='+',
                        help='List of action files')

    args = parser.parse_args()

    refs = [read_lines_from_file(args.ref_file)]
    test_set = Path(args.ref_file).name.split('.')[0]
    results = {}

    # Automatically fetch .acts files
    acts = [Path(p) for p in args.act_files]
    # unique experiments i.e. nmt and mmt for example
    exps = set([p.parent for p in acts])

    scorers = [
        AVPScorer(add_trg_eos=False),
        AVLScorer(add_trg_eos=False),
        #CWMScorer(add_trg_eos=False),
        #CWXScorer(add_trg_eos=False),
    ]

    for exp in exps:
        # get actions for this experiment
        exp_acts = [p for p in acts if p.parent == exp]
        parts = [p.name.split('.') for p in exp_acts]
        # different run prefixes
        runs = list(set([p[0] for p in parts]))
        # type of decodings i.e. wait if diff, waitk, etc.
        types = list(set([p[2] for p in parts]))

        # Evaluate baseline consecutive systems as well
        baseline_bleus = []
        for run in runs:
            hyp_fname = f'{exp}/{run}.{test_set}.gs'
            if os.path.exists(hyp_fname):
                bleu = compute_bleu(Path(hyp_fname), refs)
                baseline_bleus.append(bleu)
            else:
                baseline_bleus.append(-1)
        results[exp.name] = {m.name: '0' for m in scorers}
        results[exp.name]['Q2AVP'] = '0'
        baseline_bleus = np.array(baseline_bleus)
        results[exp.name]['BLEU'] = f'{baseline_bleus.mean():2.2f} ({baseline_bleus.std():.4f})'

        # Evaluate each decoding type and keep multiple run scores
        for typ in types:
            scores = defaultdict(list)
            for run in runs:
                act_fname = f'{exp}/{run}.{test_set}.{typ}.acts'
                hyp_fname = f'{exp}/{run}.{test_set}.{typ}.gs'

                # Compute BLEU
                bleu = compute_bleu(Path(hyp_fname), refs)
                scores['BLEU'].append(bleu)

                if os.path.exists(act_fname):
                    # Compute delay metrics
                    run_scores = [s.compute_from_file(act_fname) for s in scorers]

                    for sc in run_scores:
                        scores[sc.name].append(sc.score)

                scores['Q2AVP'] = bleu / scores['AVP'][-1]

            # aggregate
            scores = {k: np.array(v) for k, v in scores.items()}
            means = {k: v.mean() for k, v in scores.items()}
            sdevs = {k: v.std() for k, v in scores.items()}
            str_scores = {m: f'{means[m]:4.2f} ({sdevs[m]:.2f})' for m in scores.keys()}

            results[f'{exp.name}_{typ}'] = str_scores

    headers = ['Name'] + [sc.name for sc in scorers] + ['BLEU', 'Q2AVP']
    results = [[name, *[scores[key] for key in headers[1:]]] for name, scores in results.items()]
    # alphabetical sort
    results = sorted(results, key=lambda x: x[0].rsplit('_', 1)[-1])
    # print
    print(tabulate.tabulate(results, headers=headers))
