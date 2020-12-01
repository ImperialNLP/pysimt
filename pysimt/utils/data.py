from typing import List

import torch
from torch.utils.data import DataLoader


def sort_predictions(data_loader, results):
    """Recovers the dataset order when bucketing samplers are used."""
    if getattr(data_loader.batch_sampler, 'store_indices', False):
        results = [results[i] for i, j in sorted(
            enumerate(data_loader.batch_sampler.orig_idxs), key=lambda k: k[1])]
    return results


def make_dataloader(dataset, pin_memory=False, num_workers=0):
    return DataLoader(
        dataset, batch_sampler=dataset.sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory, num_workers=num_workers)



