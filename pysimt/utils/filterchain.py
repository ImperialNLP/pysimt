import re
import pathlib
from typing import List, Union

from .io import get_temp_file, fopen


class FilterChain:
    """A sequential filter chain to post-process list of tokens. The **available
    filters are:**

    `c2w`: Stitches back space delimited characters to words.
        Necessary for word-level BLEU, etc. when using char-level NMT.

    `lower`: Lowercase the input(s).

    `upper`: Uppercase the input(s).

    `de-bpe`: Stitches back `@@ ` BPE subword units.

    `de-spm`: Stitches back `▁` Sentence Piece (SPM).

    `de-segment`: Converts `<tag:morpheme>` to normal form

    `de-compound`: Stitches back German compound splittings (zmorph)

    `de-hyphen`: De-hyphenate `foo @-@ bar` constructs of Moses tokenizer.

    Args:
        filters: A list of strings representing filters to apply.

    """
    _FILTERS = {
        'de-bpe': lambda s: s.replace("@@ ", "").replace("@@", ""),
        'de-tag': lambda s: re.sub('<[a-zA-Z][a-zA-Z]>', '', s),
        # Decoder for Google sentenpiece
        # only for default params of spm_encode
        'de-spm': lambda s: s.replace(" ", "").replace("\u2581", " ").strip(),
        # Converts segmentations of <tag:morpheme> to normal form
        'de-segment': lambda s: re.sub(' *<.*?:(.*?)>', '\\1', s),
        # Space delim character sequence to non-tokenized normal word form
        'c2w': lambda s: s.replace(' ', '').replace('<s>', ' ').strip(),
        # Filters out fillers from compound splitted sentences
        'de-compound': lambda s: (s.replace(" @@ ", "").replace(" @@", "")
                                  .replace(" @", "").replace("@ ", "")),
        # de-hyphenate when -a given to Moses tokenizer
        'de-hyphen': lambda s: re.sub(r'\s*@-@\s*', '-', s),
        'lower': lambda s: s.lower(),
        'upper': lambda s: s.upper(),
    }

    def __init__(self, filters: List[str]):
        assert not set(filters).difference(set(self._FILTERS.keys())), \
            "Unknown evaluation filter given."
        self.filters = filters
        self._funcs = [self._FILTERS[k] for k in self.filters]

    def _apply(self, list_of_strs: List[str]) -> List[str]:
        """Applies filters consecutively on a list of sentences."""
        for func in self._funcs:
            list_of_strs = [func(s) for s in list_of_strs]
        return list_of_strs

    def apply(self, _input: Union[List[str], pathlib.Path]) -> List[str]:
        """Applies the filterchain on a given input.

        Args:
            _input: If `pathlib.Path` (it can also be a glob expression),
                temporary file(s) with filters applied are returned.
                If a list of sentences is given, a list of post-processed
                sentences is returned.
        """
        if isinstance(_input, pathlib.Path):
            # Need to create copies of reference files with filters applied
            # and return their paths instead
            fnames = _input.parent.glob(_input.name)
            new_fnames = []
            for fname in fnames:
                lines = []
                f = fopen(fname)
                for line in f:
                    lines.append(line.strip())
                f.close()
                f = get_temp_file()
                for line in self._apply(lines):
                    f.write(line + '\n')
                f.close()
                new_fnames.append(f.name)
            return new_fnames

        elif isinstance(_input, list):
            return self._apply(_input)

    def __repr__(self):
        return "FilterChain({})".format(" -> ".join(self.filters))
