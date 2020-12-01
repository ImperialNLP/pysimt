import logging
from typing import List

import torch
from torch.utils.data import DataLoader

from .misc import fopen, pbar

logger = logging.getLogger('pysimt')


def read_reference_files(*args) -> List[List[str]]:
    """Read every file given in `args` and produce a list of lists that
    supports multiple references."""
    all_lines = []

    for fname in args:
        lines = []
        with open(fname) as f:
            for line in f:
                lines.append(line.strip())
        all_lines.append(lines)

    ref_lens = [len(lns) for lns in all_lines]
    assert len(set(ref_lens)) == 1, \
        "Reference streams do not have the same lengths."

    return all_lines


def read_hypothesis_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines


def sort_predictions(data_loader, results):
    """Recovers the dataset order when bucketing samplers are used."""
    if getattr(data_loader.batch_sampler, 'store_indices', False):
        results = [results[i] for i, j in sorted(
            enumerate(data_loader.batch_sampler.orig_idxs), key=lambda k: k[1])]
    return results


def make_dataloader(dataset, pin_memory=False, num_workers=0):
    if num_workers != 0:
        logger.info('Forcing num_workers to 0 since it fails with torch 0.4')
        num_workers = 0

    return DataLoader(
        dataset, batch_sampler=dataset.sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory, num_workers=num_workers)


def sort_batch(seqbatch):
    """Sorts torch tensor of integer indices by decreasing order."""
    # 0 is padding_idx
    omask = (seqbatch != 0).long()
    olens = omask.sum(0)
    slens, sidxs = torch.sort(olens, descending=True)
    oidxs = torch.sort(sidxs)[1]
    return (oidxs, sidxs, slens.data.tolist(), omask.float())


def read_sentences(fname, vocab, bos=False, eos=True):
    lines = []
    lens = []
    with fopen(fname) as f:
        for idx, line in enumerate(pbar(f, unit='sents')):
            line = line.strip()

            # Empty lines will cause a lot of headaches,
            # get rid of them during preprocessing!
            assert line, "Empty line (%d) found in %s" % (idx + 1, fname)

            # Map and append
            seq = vocab.sent_to_idxs(line, explicit_bos=bos, explicit_eos=eos)
            lines.append(seq)
            lens.append(len(seq))

    return lines, lens
