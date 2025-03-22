__version__ = '1.0.0'
"""
`pysimt` is a `PyTorch`-based sequence-to-sequence (S2S) framework that facilitates
research in unimodal and multi-modal machine translation. The framework
is especially geared towards a set of recent simultaneous MT approaches, including
heuristics-based decoding and prefix-to-prefix training/decoding. Common metrics
such as average proportion (AP), average lag (AL), and consecutive wait (CW)
are provided through well-defined APIs as well.


.. include:: ./docs.md
"""


# Disable documentation generation for the following sub modules
__pdoc__ = {
    'cocoeval': False,
    'config': False,
    'logger': False,
}
