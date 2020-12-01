# -*- coding: utf-8 -*-
import math
import torch


class ScaledDotAttention(torch.nn.Module):

    def __init__(self, model_dim, n_heads, dropout=0.0):
        """
        Creates a ScaledDotAttention.
        :param model_dim: The model dimensions.
        :param n_heads: The number of heads.
        :param dropout: The dropout value. Default 0.0.
        """
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads

        self.lin_k = torch.nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.lin_q = torch.nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.lin_v = torch.nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin_o = torch.nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.head_dim = self.model_dim // self.n_heads
        self.scale = math.sqrt(self.head_dim)

    def forward(self, inputs):
        """Scaled dot-product attention forward-pass

        :param inputs: dictionary with query, key, value and mask tensors
            the shape of the tensors are (tstep, bsize, dim) except for the
            mask which is (bsize, query_len, key_len)

        :return: the output from the forward pass, the attention weights
        """
        q, k, v, mask = inputs
        q_len, _, _ = q.shape
        query, key, value = self._project_and_reshape(q, k, v)

        attn_weights = self._compute_attn_weights(query, key, mask)
        attn_probs = self.dropout(attn_weights)
        scores = torch.matmul(attn_probs, value)
        out = self.lin_o(self._view_as_concat(scores, q_len))

        return out, attn_weights

    def _project_and_reshape(self, q, k, v):
        """
        Projects the q, k and v and reshapes it into size (bsize, n_heads, q|k|v_len, head_dim).
        :param q: q of shape (q_len, b_size, model_dim)
        :param k: k of shape (k_len, b_size, model_dim)
        :param v: v of shape (v_len, b_size, model_dim)
        :return: The query, key, value of shape (b_size, n_heads, q|k|v_len, head_dim).
        """
        query = self._view_as_headed(self.lin_q(q))
        key = self._view_as_headed(self.lin_k(k))
        value = self._view_as_headed(self.lin_v(v))
        return query, key, value

    def _compute_attn_weights(self, query, key, mask):
        """
        Computes the normalized attention scores.
        :param query: The query of shape (b_size, n_heads, q_len, head_dim).
        :param key: The key of shape (b_size, n_heads, k_len, head_dim).
        :param mask: The value of shape (b_size, _, k_len).
        :return: The normalized attention scores of shape (b_size, n_heads, q_len, k_len).
        """
        attn = torch.matmul(query.div(self.scale), key.transpose(-2, -1))
        attn = self._apply_mask(mask, attn)
        return attn.softmax(dim=-1)

    def _view_as_headed(self, data):
        """
        Reshapes the data into a head format.
        :param data: (seq_len, b_size, model_dim)
        :return: (b_size, n_heads, seq_len, head_dim).
        """
        return data.view(data.shape[0], data.shape[1], self.n_heads, -1).permute(1, 2, 0, 3)

    def _view_as_concat(self, data, q_len):
        return data.permute(2, 0, 1, 3).contiguous().view(q_len, -1, self.model_dim)

    @staticmethod
    def _apply_mask(mask, attn):
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn.masked_fill_(mask, -1e8)
        return attn
