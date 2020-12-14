from typing import List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, trg_vocab_size, label_smoothing=0.1, reduction='mean', with_logits=True, ignore_index=0):
        """
        Creates a Label Smoothing Loss.
        Based on: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py#L194
        :param trg_vocab_size: The target vocabulary size.
        :param label_smoothing: The label smoothing value. Default: 0.1.
        :param reduction: The loss reduction. Default: 'mean'.
        :param with_logits: Whether the predictions are logits. Default: True.
        :param ignore_index: The value to be ignored by the loss. Can be used to ignore padding tokens. Default 0.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        self.kl_divergence = nn.KLDivLoss(reduction=reduction)

        self._create_one_hot(label_smoothing, trg_vocab_size)
        self.confidence = 1.0 - label_smoothing

    def forward(self, predictions, target):
        """
        Computes the loss.
        :param predictions: The predictions of shape (N, C) where C is the number of classes.
                            If with_logits is True, a log_softmax will be applied to obtain valid probabilities.
        :param target: The target values of shape (N).
        :return: The computed loss.
        """
        if self.with_logits is True:
            predictions = F.log_softmax(predictions, dim=-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        self._apply_mask(model_prob, target)

        return self.kl_divergence(predictions, model_prob)

    def _create_one_hot(self, label_smoothing, trg_vocab_size):
        smoothing_value = label_smoothing / (trg_vocab_size - 2)
        one_hot = torch.full((trg_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def _apply_mask(self, model_prob, target):
        mask = (target == self.ignore_index).unsqueeze(1)
        model_prob.masked_fill_(mask, 0)


def get_activation_fn(name: Optional[str]):
    """Returns a callable activation function from `torch`."""
    if name in (None, 'linear'):
        return lambda x: x
    elif name in ('sigmoid', 'tanh'):
        return getattr(torch, name)
    else:
        return getattr(F, name)


def generate_default_mask(data, dim1=None):
    """
    Returns a default mask which allows the model to attend over all positions.
    :param data: The data of shape (sequence_len, batch_size)
    :param dim1: The first dimension of the mask. If none, it is equal to sequence_len.
    :return:
    """
    batch_size = data.size(1)
    sequence_len = data.size(0)
    if dim1 is None:
        dim1 = sequence_len
    return torch.zeros(batch_size, dim1, sequence_len).bool().to(data.device)


def generate_visual_features_padding_masks(data, pad_value=0):
    """
    Returns a mask based on the data. For values of the padding token=0, the mask will contain 1, indicating the
    model cannot attend over these positions.
    :param data: The data of shape (sequence_len, batch_size, feature_dim)
    :param pad_value: The value of the padding. Default: 0.
    :return: The respective mask of shape (batch_size, 1, sequence_len)
    """
    with torch.no_grad():
        return (data == pad_value).all(dim=-1).t().to(data.device).unsqueeze(1)


def generate_padding_masks(data, pad_value=0):
    """
    Returns a mask based on the data. For values of the padding token=0, the mask will contain 1, indicating the
    model cannot attend over these positions.
    :param data: The data of shape (sequence_len, batch_size)
    :param pad_value: The value of the padding. Default: 0.
    :return: The respective mask of shape (batch_size, 1, sequence_len)
    """
    with torch.no_grad():
        mask = (data == pad_value).to(data.device).t().unsqueeze(1)
    return mask


def generate_lookahead_mask(data, k=1, dim1=None):
    """
    Generates a lookahead mask, preventing the decoder from attending to previous positions when computing the
    attention. The mask will contain 1 for positions which should not be attended to.
    :param data: The data of shape (sequence_len, batch_size).
    :param k: The offset for the lookahead mask. By default it's 0. Example: In the decoder self-attention, each decoder
              word can use only itself and all previous words.
    :param dim1: The first dimension of the mask. If none, it is equal to sequence_len.
    :return: The lookahead mask of shape (1, dim1, sequence_len)
    """
    sequence_len = data.size(0)
    if dim1 is None:
        dim1 = sequence_len

    lookahead_mask = torch.triu(torch.ones((1, dim1, sequence_len)), diagonal=k)

    return lookahead_mask.to(data.device).bool()


def generate_combined_mask(data, k=1):
    """
    Generates a combined padding and lookahead mask.
    The mask will contain 1 for positions which should not be attended to.
    :param data: The data of shape (sequence_len, batch_size).
    :param k: The offset for the lookahead mask. By default it's 1, allowing the decoder to observe the <bos> token.
    :return: Combined padding and lookahead mask.
    """
    padding_mask = generate_padding_masks(data)
    lookahead_mask = generate_lookahead_mask(data, k)
    combined_mask = padding_mask | lookahead_mask

    return combined_mask


def readable_size(n: int) -> str:
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)


def get_module_groups(layer_names: List[str]) -> List[str]:
    groups = set()
    for name in layer_names:
        if '.weight' in name:
            groups.add(name.split('.weight')[0])
        elif '.bias' in name:
            groups.add(name.split('.bias')[0])
    return sorted(list(groups))


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return "# parameters: {} ({} learnable)".format(
        readable_size(n_param_all), readable_size(n_param_learnable))
