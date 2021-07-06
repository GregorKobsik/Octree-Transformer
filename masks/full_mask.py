import torch


def full_mask(seq_len, device=None):
    """ Creates a full mask, which allows to access to all tokens. """
    return torch.full((seq_len, seq_len), 0, device=device)
