from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ObjectDetectionsDataset(Dataset):
    r"""A PyTorch dataset for .npz key-value stores for `Bottom-up-Top-Down (BUTD)`
    object detection model.

    Arguments:
        fname (str or Path): A string or ``pathlib.Path`` object for
            the relevant numpy file.
        revert (bool, optional): If `True`, the data order will be reverted
            for adversarial/incongruent experiments during test-time.
        feat_mode (str, optional): What type of features to return for a given
            sample. Default is `roi_feats` which will yield `num_proposals`
            feature vectors per image. `None` defaults to `roi_feats` as well.
        num_regions (int, optional): How many proposals to keep. The default
            is `36`.
    """

    def __init__(self, fname, revert=False, feat_mode='roi_feats', num_regions=36, **kwargs):
        self.path = Path(fname)
        self.feat_mode = feat_mode
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        # If None, default to `roi_feats`
        if self.feat_mode is None:
            self.feat_mode = 'roi_feats'

        # open the file
        self._handle = np.load(self.path)

        self.order = list(range(self._handle['order'].size))
        if revert:
            self.order = self.order[::-1]

        # Dataset size
        self.size = len(self.order)

        # Handle feat_mode
        if self.feat_mode == 'roi_feats':
            data = torch.from_numpy(
                self._handle['features'].astype(np.float32))
        elif self.feat_mode == 'objs':
            # Integer ids for objects: (n, num_regions, 1)
            data = torch.from_numpy(
                self._handle['objects_id'].astype(np.int32)).long().unsqueeze(-1)
        elif self.feat_mode == 'attrs+objs':
            # Integer ids for attributes and objects: (n, num_regions, 2)
            objs = torch.from_numpy(self._handle['objects_id'].astype(np.int32))
            attr = torch.from_numpy(self._handle['attrs_id'].astype(np.int32))
            data = torch.stack((objs, attr), -1).long()
        elif self.feat_mode == 'roi_feats+boxes':
            roi_feats = torch.from_numpy(self._handle['features'].astype(np.float32))
            boxes = torch.from_numpy(self._handle['boxes_and_areas'].astype(np.float32))
            data = torch.cat((roi_feats, boxes), -1)
        else:
            raise RuntimeError(f'feat_mode={self.feat_mode!r} is not known!')

        # check size
        assert data.shape[0] == self.size, ".npz file's 1-dim is not sample"
        self.data = data[:, :num_regions, :]

    @staticmethod
    def to_torch(batch, **kwargs):
        """Assumes x.shape == (n, *)."""
        return torch.stack(batch, dim=0).permute(1, 0, 2)

    def __getitem__(self, idx):
        return self.data[self.order[idx]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples, feat_mode: {})\n".format(
            self.__class__.__name__, self.path.name, self.__len__(), self.feat_mode)
        return s
