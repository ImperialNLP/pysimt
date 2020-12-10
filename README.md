![pysimt](https://github.com/ImperialNLP/pysimt/blob/master/logo.png?raw=true "pysimt")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

This repository includes the codes, the experiment configurations and the scripts
to prepare/download data for the [Simultaneous Machine Translation with Visual Context](https://www.aclweb.org/anthology/2020.emnlp-main.184.pdf)
paper presented at EMNLP 2020.


## Overview

`pysimt` is a `PyTorch`-based sequence-to-sequence framework that facilitates
research in unimodal and multi-modal machine translation. The framework
is especially geared towards a set of recent simultaneous MT approaches, including
heuristics-based decoding and prefix-to-prefix training/decoding. Common metrics
such as average proportion (AP), average lag (AL), and consecutive wait (CW)
are provided through well-defined APIs as well.

Please visit [https://imperialnlp.github.io/pysimt](https://imperialnlp.github.io/pysimt)
for detailed documentation.


## Citation

```
@inproceedings{caglayan-etal-2020-simultaneous,
    title = "Simultaneous Machine Translation with Visual Context",
    author = {Caglayan, Ozan  and
      Ive, Julia  and
      Haralampieva, Veneta  and
      Madhyastha, Pranava  and
      Barrault, Lo{\"\i}c  and
      Specia, Lucia},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.184",
    pages = "2350--2361",
}
```

## Installation
The essential dependency of `pysimt` is `torch>=1.7`. The following command
will create an appropriate Anaconda environment with `pysimt` installed within in editable mode.

```bash
conda env create -f environment.yml
```

Once the installation is done, run `pysimt-install-extra` command if you want
to use METEOR as an evaluation metric.
