import torch


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
