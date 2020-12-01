import math

import numpy as np

import torch
from torch import nn


class MultiheadAttention(nn.Module):
    """General purpose multihead attention implementation."""
    def __init__(self, input_dim, proj_dim, n_heads=1, dropout=0.0,
                 attn_type='cross', initializer='xavier_uniform'):
        assert proj_dim % n_heads == 0, "proj_dim not divisible by n_heads."
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.n_heads = n_heads
        self.head_dim = self.proj_dim // self.n_heads
        self.scale = math.sqrt(self.head_dim)
        self.minus_inf = float('-inf')
        self.attn_type = attn_type
        self.initializer = initializer
        self.p_dropout = dropout

        self._apply_projections_and_reshape = getattr(
            self, f'_apply_projections_and_reshape_{self.attn_type}')

        # dropout over attention probability
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else lambda x: x

        self._create_layers()
        self._reset_parameters(getattr(nn.init, f'{initializer}_'))

    def __repr__(self):
        s = f"MultiheadAttention({self.input_dim} -> {self.proj_dim}, {self.n_heads} heads, "
        s += f"type={self.attn_type!r}, dropout={self.p_dropout})"
        return s

    def view_as_headed(self, x):
        """Returns a view of shape `[bsz, n_heads, seq_len, head_dim]`
        from `[bsz, seq_len, head_dim * n_heads]`."""
        return x.view(x.size(0), x.size(1), self.n_heads, -1).transpose(1, 2)

    @staticmethod
    def view_as_concat(x):
        """Returns a view of shape `[bsz, seq_len, head_dim * n_heads]`
        from `[bsz, n_heads, seq_len, head_dim]`."""
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1)

    def _reset_parameters(self, init_fn):
        """Reinitializes layer weights."""
        for param in self.parameters():
            init_fn(param)

    def _create_layers(self):
        """Create projection layer weights."""
        self.lin_o = nn.Parameter(torch.Tensor(self.proj_dim, self.proj_dim))
        if self.attn_type != 'self':
            self.lin_k = nn.Parameter(torch.Tensor(self.input_dim, self.proj_dim))
            self.lin_q = nn.Parameter(torch.Tensor(self.input_dim, self.proj_dim))
            self.lin_v = nn.Parameter(torch.Tensor(self.input_dim, self.proj_dim))
        else:
            self.lin_k = nn.Parameter(torch.Tensor(self.input_dim, 3 * self.proj_dim))

    def _apply_projections_and_reshape_self(self, k, v=None, q=None):
        """Projects key, value and queries and returns multi-head view
        for self-attention variant.

        Args:
            k: Tensor of shape `[batch_size, v_len, dim]`.
            v: `None` for self-attention. This is not used.
            q: `None` for self-attention. This is not used.

        Returns:
            A tuple of 3 tensors for k,v,q projections, each with shape
            `[batch_size, n_heads, v_len, head_dim]`.
        """
        return (
            self.view_as_headed(t) for t in k.matmul(self.lin_k).chunk(3, dim=-1))

    def _apply_projections_and_reshape_cross(self, k, v, q):
        """Projects key, value and queries and returns multi-head view
        for cross-attention variant.

        Args:
            k: Tensor of shape `[batch_size, v_len, dim]`.
            v: Tensor of shape `[batch_size, v_len, dim]`.
            q: Tensor of shape `[batch_size, q_len, dim]`.

        Returns:
            A tuple of 3 tensors for k,v,q projections, each with shape
            `[batch_size, n_heads, (v|q)_len, head_dim]`.
        """
        return (self.view_as_headed(k.matmul(self.lin_k)),
                self.view_as_headed(v.matmul(self.lin_v)),
                self.view_as_headed(q.matmul(self.lin_q)))

    def _compute_scores(self, query, key, k_mask=None):
        """Computes normalized scaled dot-product scores between query and key.

        Args:
            query: Tensor of shape `[batch_size, n_heads, q_len, dim]`.
            key: Tensor of shape `[batch_size, n_heads, v_len, dim]`.
            k_mask: Tensor of shape `[batch_size, v_len]`.

        Returns:
            Tensor of shape `[batch_size, n_heads, q_len, v_len]` with
                normalized attention weights.
        """
        scores = torch.matmul(query.div(self.scale), key.transpose(-2, -1))
        if k_mask is not None:
            # mask <pad>'ded positions
            scores.masked_fill_(k_mask[:, None, None, :], self.minus_inf)
        return self.dropout(scores.softmax(dim=-1))

    def _apply_scores(self, p, value, q_mask=None):
        """Applies normalized attention weights on `value`. `q_mask`
        is used to zero padded positions afterwards.

        Args:
            p: Tensor of shape `[batch_size, n_heads, q_len, v_len]`.
            value: Tensor of shape `[batch_size, n_heads, v_len, dim]`.
            q_mask: Tensor of shape `[batch_size, q_len]`.

        Returns:
            Tensor of shape `[batch_size, n_heads, v_len, dim]`.
        """
        ctx = torch.matmul(p, value)
        if q_mask is not None:
            # zero out <pad>'ded positions
            ctx.mul_(q_mask[:, None, :, None].logical_not())
        return ctx

    def forward(self, k, v=None, q=None, k_mask=None, q_mask=None):
        kp, vp, qp = self._apply_projections_and_reshape(k, v, q)

        # Get normalized scores
        alpha = self._compute_scores(qp, kp, k_mask)

        # Get weighted contexts for each head -> concat -> project
        return self.view_as_concat(
            self._apply_scores(alpha, vp, q_mask)).matmul(self.lin_o)


def get_upstream_impl(dim, n_heads):
    mha = nn.MultiheadAttention(dim, n_heads, bias=False)
    nn.init.eye_(mha.out_proj.weight.data)
    list(map(lambda i: nn.init.eye_(i), mha.in_proj_weight.data.chunk(3, dim=0)))
    nn.init.eye_(mha.in_proj_weight.data[:dim])
    nn.init.eye_(mha.in_proj_weight.data[dim:2*dim])
    nn.init.eye_(mha.in_proj_weight.data[-dim:])
    return mha


def get_own_self_impl(i_dim, p_dim, n_heads):
    self_att = MultiheadAttention(input_dim=i_dim, proj_dim=p_dim, n_heads=n_heads, attn_type='self')
    print(self_att)
    nn.init.eye_(self_att.lin_o.data)
    list(map(lambda x: nn.init.eye_(x), self_att.lin_k.data.chunk(3, dim=-1)))
    return self_att


def get_own_cross_impl(i_dim, p_dim, n_heads):
    cross_att = MultiheadAttention(input_dim=i_dim, proj_dim=p_dim, n_heads=n_heads)
    print(cross_att)
    nn.init.eye_(cross_att.lin_o.data)
    nn.init.eye_(cross_att.lin_k.data)
    nn.init.eye_(cross_att.lin_q.data)
    nn.init.eye_(cross_att.lin_v.data)
    return cross_att


def main():
    np.random.seed(2)
    torch.manual_seed(3)
    torch.cuda.manual_seed(4)

    input_dim = 512
    batch_size = 100
    vocab_size = 1000

    # Create the embeddings
    embs = nn.Embedding(vocab_size, embedding_dim=input_dim, padding_idx=0)

    # Sample sequence lengths
    src_seq_lens = np.random.normal(6, 1, size=(batch_size,)).astype('int')
    trg_seq_lens = np.random.normal(6, 1, size=(batch_size,)).astype('int')

    # Sample random vocab IDs
    src_idxs = torch.randint(
        low=1, high=vocab_size, size=(batch_size, src_seq_lens.max()))
    trg_idxs = torch.randint(
        low=1, high=vocab_size, size=(batch_size, trg_seq_lens.max()))

    # pad short sequences
    for seq, seqlen in enumerate(src_seq_lens):
        src_idxs[seq, seqlen:].fill_(0)

    for seq, seqlen in enumerate(trg_seq_lens):
        trg_idxs[seq, seqlen:].fill_(0)

    # masks with `True` for padded positions
    src_padding_mask = src_idxs.eq(0)
    trg_padding_mask = trg_idxs.eq(0)

    # Verify lengths
    assert np.allclose(src_seq_lens, src_idxs.ne(0).sum(1))
    assert np.allclose(trg_seq_lens, trg_idxs.ne(0).sum(1))

    # get embeddings
    x = embs(src_idxs)
    y = embs(trg_idxs)

    # Verify lengths using embeddings
    assert np.allclose(src_seq_lens, x.sum(-1).ne(0.0).sum(1))
    assert np.allclose(trg_seq_lens, y.sum(-1).ne(0.0).sum(1))

    mha = get_upstream_impl(input_dim, 1)
    xp = x.transpose(0, 1)
    yp = y.transpose(0, 1)

    h_mha_self, p_mha_self = mha(
        query=xp, key=xp, value=xp, key_padding_mask=src_padding_mask)
    h_mha_cross, p_mha_cross = mha(
        query=yp, key=xp, value=xp, key_padding_mask=src_padding_mask)
    h_mha_self.transpose_(0, 1)
    h_mha_cross.transpose_(0, 1)

    # self attention
    # q_mask: src
    self_att = get_own_self_impl(input_dim, input_dim, n_heads=1)
    h_self = self_att(k=x, v=x, q=x, k_mask=src_padding_mask, q_mask=None)

    assert torch.allclose(h_self, h_mha_self, atol=1e-1)

    # self attention with identity projections should produce the query itself
    assert torch.allclose(
        self_att(x, x, x, src_padding_mask, src_padding_mask), x, atol=1e-1)

    # cross attention
    # q_mask: trg
    cross_att = get_own_cross_impl(input_dim, input_dim, n_heads=1)
    h_cross = cross_att(k=x, v=x, q=y, k_mask=src_padding_mask, q_mask=trg_padding_mask)

    assert torch.allclose(
        cross_att(x, x, y, src_padding_mask, None), h_mha_cross, atol=1e-1)

    #################
    # multi-head test
    #################
    for nh in (1, 2, 4, 8, 16, 32):
        print(f'# heads: {nh}')
        self_att = get_own_self_impl(input_dim, input_dim, n_heads=nh)
        cross_att = get_own_cross_impl(input_dim, input_dim, n_heads=nh)
        torc_att = get_upstream_impl(input_dim, nh)

        h_torc, p_torc = torc_att(xp, xp, xp, key_padding_mask=src_padding_mask)
        h_torc.transpose_(0, 1)

        h_self = self_att(k=x, k_mask=src_padding_mask, q_mask=None)
        h_cross = cross_att(x, x, x, k_mask=src_padding_mask, q_mask=None)

        assert torch.allclose(h_self, h_torc, atol=1e-1)
        assert torch.allclose(h_cross, h_torc, atol=1e-1)

    self_att = get_own_self_impl(input_dim, 256, n_heads=2)
    cross_att = get_own_cross_impl(input_dim, 256, n_heads=2)
    h_self = self_att(k=x, k_mask=src_padding_mask, q_mask=None)
    h_cross = cross_att(x, x, x, k_mask=src_padding_mask, q_mask=None)


if __name__ == '__main__':
    main()
