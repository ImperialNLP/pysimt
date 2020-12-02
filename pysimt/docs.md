Features
--------

`pysimt` includes two state-of-the-art S2S approaches to neural machine
translation (NMT) to begin with:

* [RNN-based attentive NMT (Bahdanau et al. 2014)]
* [Self-attention based Transformers NMT (Vaswani et al. 2017)]

[RNN-based attentive NMT (Bahdanau et al. 2014)]:
  http://arxiv.org/pdf/1409.0473

[Self-attention based Transformers NMT (Vaswani et al. 2017)]:
  http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

The toolkit mostly emphasizes on multimodal machine translation (MMT), and
therefore the above models easily accomodate multiple source modalities
through [encoder-side (Caglayan et al. 2020)] and [decoder-side multimodal attention (Caglayan et al. 2016)] approaches.

[encoder-side (Caglayan et al. 2020)]:
  https://www.aclweb.org/anthology/2020.emnlp-main.184

[decoder-side multimodal attention (Caglayan et al. 2016)]:
  http://arxiv.org/pdf/1609.03976

### Simultaneous NMT

The following notable approaches in the simultaneous NMT field are implemented:

* [Heuristics-based decoding approaches wait-if-diff and wait-if-worse (Cho and Esipova, 2016)]
* [Prefix-to-prefix training and decoding approach wait-k (Ma et al., 2019)]

[Heuristics-based decoding approaches wait-if-diff and wait-if-worse (Cho and Esipova, 2016)]:
  https://arxiv.org/pdf/1606.02012

[Prefix-to-prefix training and decoding approach wait-k (Ma et al., 2019)]:
  https://www.aclweb.org/anthology/P19-1289.pdf


### Simultaneous MMT

The toolkit includes the reference implementation for the following conference
papers that initiated research in Simultaneous MMT:

* [Simultaneous Machine Translation with Visual Context (Caglayan et al. 2020)]
* [Towards Multimodal Simultaneous Neural Machine Translation (Imankulova et al. 2020)]

[Simultaneous Machine Translation with Visual Context (Caglayan et al. 2020)]:
  https://www.aclweb.org/anthology/2020.emnlp-main.184.pdf

[Towards Multimodal Simultaneous Neural Machine Translation (Imankulova et al. 2020)]:
  http://statmt.org/wmt20/pdf/2020.wmt-1.70.pdf

### Other features

* CPU / (Single) GPU training of sequence-to-sequence frameworks
* Reproducible experimentation through well-defined configuration files
* Easy multimodal training with parallel corpora
* Logging training progress and validation performance to Tensorboard
* Text, Image Features and Speech encoders
* Early-stopping and model checkpointing using various criteria such as MultiBLEU,
  SacreBLEU, METEOR, word error rate (WER), character error rate (CER), etc.
* Ready-to-use latency metrics for simultaneous MT, including average proportion (AP),
  average lag (AL), and consecutive wait (CW)
* Beam search translation for consecutive, greedy search translation for simultaneous MT
* Utilities to produce reports for simultaneous MT performance


Installation
------------
Essentially, `pysimt` requires `Python>=3.7` and `torch>=1.7.0`. You can access
the other dependencies in the provided `environment.yml` file.

The following command will create an appropriate Anaconda environment with `pysimt`
installed in editable mode. This will allow you to modify to code in the GIT
checkout folder, and then run the experiments directly.

    $ conda env create -f environment.yml

.. note::
  If you want to use the METEOR metric for early-stopping or the `pysimt-coco-metrics`
  script to evaluate your models' performance, you need to run the `pysimt-install-extra`
  script within the **pysimt** Anaconda environment. This will download and install
  the METEOR paraphrase files under the `~/.pysimt` folder.


Command-line tools
------------------
Once installed, you will have access to three command line utilities:

### pysimt-build-vocab

* Since `pysimt` does not pre-process, tokenize, segment the given text files
automagically, all these steps should be done by the user prior to training, and
the relevant vocabulary files should be constructed using `pysimt-build-vocab`.
* Different vocabularies should be constructed for source and target language
representations (unless `-s` is given).
* The resulting files are in `.json` format.

**Arguments:**

* `-o, --output-dir OUTPUT_DIR`: Output directory where the resulting vocabularies will be stored.
* `-s, --single`: If given, a single vocabulary file for all the
  given training corpora files will be constructed. Useful for weight tying in embedding layers.
* `-m, --min-freq`: If given an integer `M`, it will filter out tokens occuring `< M` times.
* `-M, --max-items`: If given an integer `M`, the final vocabulary will be limited to `M` most-frequent tokens.
* `-x, --exclude-symbols`: If given, the vocabulary will **not include special markers** such as `<bos>, <eos>`.
  This should be used cautiously, and only for ad-hoc model implementations, as it may break the default models.
* `files`: A variable number of training set corpora can be provided. If `-s` is not
  given, one vocabulary for each will be created.

### pysimt-coco-metrics
This is a simple utility that computes BLEU, METEOR, CIDEr, and ROUGE-L
using the well known [coco-caption] library. The library is shipped within
`pysimt` so that you do not have to install it separately.

**Arguments:**

* `-l, --language`: If given a string `L`, the METEOR will be informed with that information.
  For languages not supported by METEOR, English will be assumed.
* `-w, --write`: For every hypothesis file given as argument, a `<hypothesis file>.score` file
  will be created with the computed metrics inside for convenience.
* `-r, --refs`: List of reference files for evaluation. The number of lines across multiple
  references should be **equal**.
* `systems`: A variable number of hypotheses files that represent system outputs.

**Example:**

    $ pysimt-coco-metrics -l de system1.hyps system2.hyps -r ref1
    $ pysimt-coco-metrics -l de system1.hyps system2.hyps -r ref1 ref2 ref3

[coco-caption]: https://github.com/tylin/coco-caption

.. note::
  This utility requires **tokenized** hypotheses and references, as further
  tokenization is not applied by the internal metrics. Specifically for BLEU,
  if you are not evaluating your models for MMT or image captioning,
  you may want to use `sacreBLEU` for detokenized hypotheses and references.

.. tip::
  The `Bleu_4` produced by this utility is equivalent to the output of
  `multi-bleu.perl` and `sacreBLEU` (when `--tokenize none` is given to the latter).

### pysimt

This is the main entry point to the software. It supports two modes, namely
**pysimt train** and **pysimt translate**.

#### Training a model

#### Translating with a pre-trained model




Configuring an experiment
--------------------------

Models
------

* A `pysimt` model derives from `torch.nn.Module` and implements specific API methods.


Contributing
------------
`pysimt` is [on GitHub]. Bug reports and pull requests are welcome.

[on GitHub]: https://github.com/ImperialNLP/pysimt


Citing the toolkit
------------------

As of now, you can cite the [following work] if you use this toolkit. We will
update this section if the software paper is published elsewhere.

[following work]:
  https://www.aclweb.org/anthology/2020.emnlp-main.184

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




License
-------
`pysimt` uses MIT License.

```
Copyright (c) 2020 NLP@Imperial

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
