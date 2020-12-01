from torch import nn
import torch
import torch.nn.functional as F


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
