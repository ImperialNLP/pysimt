"""
A dataset in `pysimt` inherits from `torch.nn.Dataset` and is designed
to read and expose a specific type of corpus.

* A dataset class name should end with the `Dataset` suffix.
* The `__init__` method should include `**kwargs` for other possible arguments.
* The `__getitem__` and `__len__` methods should be implemented.
* A static method `to_torch(batch, **kwargs)` is automatically used when
  preparing the batch tensor during forward-pass.

Please see `pysimt.datasets.TextDataset` to get an idea on how to implement
a new dataset.

"""

from .numpy import NumpyDataset
from .text import TextDataset
from .objdet import ObjectDetectionsDataset


# Second the selector function
def get_dataset(type_):
    return {
        'numpy': NumpyDataset,
        'text': TextDataset,
        'objectdetections': ObjectDetectionsDataset,
    }[type_.lower()]


# Should always be at the end
from .multimodal import MultimodalDataset   # noqa
