import bz2
import gzip
import lzma
import pathlib

from collections import deque

import numpy as np
from tqdm import tqdm


from ..cleanup import cleanup


class FileRotator:
    """A fixed queue with Path() elements where pushing a new element pops
    the oldest one and removes it from disk.

    Arguments:
        maxlen(int): The capacity of the queue.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.elems = deque(maxlen=self.maxlen)

    def push(self, elem):
        if len(self.elems) == self.maxlen:
            # Remove oldest item
            popped = self.elems.pop()
            if popped.exists():
                popped.unlink()

        # Add new item
        self.elems.appendleft(elem)

    def __repr__(self):
        return self.elems.__repr__()


def fopen(filename: str, key: str = None):
    """gzip,bzip2,xz,numpy aware file opening function."""
    assert '*' not in str(filename), "Glob patterns not supported in fopen()"

    filename = str(pathlib.Path(filename).expanduser())
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    elif filename.endswith(('.xz', '.lzma')):
        return lzma.open(filename, 'rt')
    elif filename.endswith(('.npy', '.npz')):
        if filename.endswith('.npz'):
            assert key is not None, "No key= given for .npz file."
            return np.load(filename)[key]
        else:
            return np.load(filename)
    else:
        # Plain text
        return open(filename, 'r')


def read_hypothesis_file(fname: str) -> List[str]:
    """Reads lines from a text file and returns it as a list of strings."""
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines


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


def get_temp_file(delete=False, close=False):
    """Creates a temporary file under a folder."""
    root = pathlib.Path(os.environ.get('NMTPY_TMP', '/tmp'))
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    prefix = str(root / "pysimt_{}".format(os.getpid()))
    t = tempfile.NamedTemporaryFile(
        mode='w', prefix=prefix, delete=delete)
    cleanup.register_tmp_file(t.name)
    if close:
        t.close()
    return t


def pbar(iterator, unit='it'):
    return tqdm(iterator, unit=unit, ncols=70, smoothing=0)

