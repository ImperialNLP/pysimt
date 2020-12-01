import numpy as np

from .metric import Metric


class AVPScorer:
    """Average Proportion metric (Cho and Esipova, 2016)"""
    def __init__(self, add_trg_eos=True):
        self.name = 'AVP'
        self.add_trg_eos = add_trg_eos

    def compute(self, actions):
        """`actions` is a list of strings where each string is a space-separated
        sequence of 0s/1s or R/Ws for READs and WRITEs, respectively."""
        delays = self.compute_delays(self.__process_actions(actions))
        return Metric(self.name, delays.mean(), higher_better=False)

    def __process_actions(self, actions):
        """Map R/Ws to 0/1s if any, convert each to numpy array."""
        # '3' could be used as padding, just remove it
        _lut = {ord('R'): '0', ord('W'): '1', ord('3'): None}
        if self.add_trg_eos:
            return [np.array(a.strip().translate(_lut).strip().split() + ['1'], dtype='int') for a in actions]
        else:
            return [np.array(a.strip().translate(_lut).strip().split(), dtype='int') for a in actions]

    def compute_delays(self, np_actions):
        """Compute delay per each sequence and return with averaging."""
        delays = []
        for acts in np_actions:
            len_y = acts.sum()
            len_x = acts.size - len_y
            nom = np.sum(np.cumsum(1 - acts) * acts)
            delays.append(nom / (len_x * len_y))
        return np.array(delays)

    def compute_from_file(self, fname):
        """`fname` is a text file where each line is a space separated
        sequence with either 0/1s or R/Ws."""
        actions = []
        with open(fname) as f:
            for line in f:
                actions.append(line.strip())
        return self.compute(actions)


class CWMScorer(AVPScorer):
    """Mean consecutive wait metric (Gu et al., 2017)"""
    def __init__(self, add_trg_eos=True):
        self.name = 'CWM'
        self.add_trg_eos = add_trg_eos

    def compute_sequence_cw(self, actions):
        a = (1 - actions).cumsum() * actions
        # remove 0s
        a = a[a > 0]
        # compute CW
        cw = a - np.pad(a, pad_width=(1, 0))[:-1]
        return cw[cw > 0]

    def compute_delays(self, np_actions):
        """Compute average CW per sequence."""
        cws = [self.compute_sequence_cw(act).mean() for act in np_actions]
        return np.array(cws)


class CWXScorer(CWMScorer):
    """Average maximum consecutive wait metric (Gu et al., 2017)"""
    def __init__(self, add_trg_eos=True):
        self.name = 'CWX'
        self.add_trg_eos = add_trg_eos

    def compute_delays(self, np_actions):
        """Compute average of max CW per sequence."""
        cws = [self.compute_sequence_cw(act).max() for act in np_actions]
        return np.array(cws)


class AVLScorer(AVPScorer):
    """Average Lagging metric (Ma et al., 2019)"""
    def __init__(self, add_trg_eos=True):
        self.name = 'AVL'
        self.add_trg_eos = add_trg_eos

    def compute_delays(self, np_actions):
        """Compute lag per sequence."""
        lags = []
        for acts in np_actions:
            len_y = acts.sum()
            len_x = acts.size - len_y
            ratio = len_y / len_x

            # cutoff point where reading ends
            cutoff = acts[:np.argwhere(acts == 0).flatten()[-1] + 2].sum()

            # 2nd term in eq. 8
            t2 = np.arange(cutoff) / ratio

            # compute cumulative reads and filter out zero's
            cum_reads = (1 - acts).cumsum() * acts
            cum_reads = cum_reads[cum_reads > 0]

            lags.append((cum_reads[:cutoff] - t2).mean())
        return np.array(lags)
