import torch


class Pool(torch.nn.Module):
    """A convenience layer to apply various sorts of pooling to a
    sequential tensor. The pooling operation can be `last`, `mean`, `max`, or
    `sum`.

    Args:
        operation: The pooling operator.
            It should be one from `last`, `mean`, `max`, `sum`.
        pool_dim: The dimension along which the pooling will be applied
        keepdim: Passed along to the underlying `torch` functions for
            `max`, `mean` and `sum` variants.

    Examples:
        >>> import torch
        >>> from pysimt.layers import Pool
        >>> x = torch.rand(10, 32, 200) # n_timesteps, n_samples, feat_dim
        >>> p = Pool('sum', 0)
        >>> torch.equal(p(x), x.sum(0, keepdim=True))
        True
        >>> p = Pool('max', 0)
        >>> torch.equal(p(x), x.max(0, keepdim=True)[0])
        True
        >>> p = Pool('mean', 0)
        >>> torch.equal(p(x), x.mean(0, keepdim=True))
        True
        >>> p = Pool('last', 0)
        >>> torch.equal(p(x), x.select(0, -1).unsqueeze(0))
        True
        >>> torch.equal(p(x), x[-1].unsqueeze(0))
        True
        >>> p = Pool('last', 1)
        >>> torch.equal(p(x), x.select(1, -1).unsqueeze(0))
        True
    """
    def __init__(self, operation, pool_dim, keepdim=True):
        """"""
        super().__init__()

        self.operation = operation
        self.pool_dim = pool_dim
        self.keepdim = keepdim

        assert self.operation in ["last", "mean", "max", "sum"], \
            "Pool() operation should be mean, max, sum or last."

        # Assign the shortcut
        self.forward = getattr(self, '_{}'.format(self.operation))

    def _last(self, x: torch.Tensor) -> torch.Tensor:
        return x.select(self.pool_dim, -1).unsqueeze(0)

    def _max(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.pool_dim, keepdim=self.keepdim)[0]

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.pool_dim, keepdim=self.keepdim)

    def _sum(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=self.pool_dim, keepdim=self.keepdim)

    def __repr__(self):
        return "Pool(operation={}, pool_dim={}, keepdim={})".format(
            self.operation, self.pool_dim, self.keepdim)
