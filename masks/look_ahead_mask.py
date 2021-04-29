import torch


def look_ahead_mask(seq_len, device=None):
    """ Creates a diagonal mask, which prevents the self-attention to look ahead. """
    attn_mask = torch.full((seq_len, seq_len), -float("Inf"), device=device)
    return torch.triu(attn_mask, diagonal=1)
