class Task:
    """A class which represents a task.

    Arguments:
        name(str): The task name as defined in the configuration file.
        params(dict, optional): If not `None`, stores the task-relevant
            arguments as well.

    """
    def __init__(self, name, params=None):
        assert name.count(':') == name.count('-') == 1, \
            "Task name should be in <prefix>:<src1,...,srcN>-<trg1,...,trgN> format"

        self.name = name
        self.params = params
        self.prefix, topology = self.name.split(':')
        srcs, trgs = topology.split('-')
        self.sources = srcs.split(',')
        self.targets = trgs.split(',')
        self.multi_source = len(self.sources) > 1
        self.multi_target = len(self.targets) > 1

        self.src, self.trg = None, None

        if not self.multi_source:
            self.src = self.sources[0]

        if not self.multi_target:
            self.trg = self.targets[0]

    def __repr__(self):
        return f'Task({self.name}, srcs={self.sources!r}, trgs={self.targets!r})'


class Batch:
    r"""A custom batch object used through data representation.

    Args:
        data(dict): A dictionary with keys representing the data sources
            as defined by the configuration file and values being the
            already collated `torch.Tensor` objects for a given batch.

        task(str): The task name associated with the batch.

        device(`torch.device`, optional): If given, the batch will be moved
            to the device right after instance creation.

        non_blocking(bool, optional): If `True`, the tensor copies will
            use the `non_blocking` argument.

    Notes:
        In the current code, moving to an appropriate device is handled
        by an explicit call to `.to()` method from the main loop or
        other relevant places.

    Returns:
        a `Batch` instance indexable with bracket notation. The batch size
        is accessible through the `.size` attribute.

    """
    def __init__(self, data, task, device=None, non_blocking=False):
        if device:
            self.data = {k: v.to(device, non_blocking=non_blocking) for k, v in data.items()}
        else:
            self.data = data

        self.task = Task(task)

        dim1s = set([x.size(1) for x in data.values()])
        assert len(dim1s) == 1, \
            "Incompatible batch dimension (1) between modalities."
        self.size = dim1s.pop()

    def __getitem__(self, key):
        return self.data[key]

    def to(self, device, non_blocking=False):
        self.data.update(
            {k: v.to(device, non_blocking=non_blocking) for k, v in self.data.items()})

    def __repr__(self):
        s = f"Batch(size={self.size}, task={self.task})\n"
        for key, value in self.data.items():
            s += f"  {key:10s} -> {value.shape} - {value.device}\n"
        return s[:-1]
