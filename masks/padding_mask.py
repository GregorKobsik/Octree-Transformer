import torch


def padding_mask(input_sequence, device=None):
    """ Create a padding mask for the given input.

        Always assumens '0' as a padding value. `input_sequence` has the shape (N, S).
    """
    return torch.zeros_like(input_sequence, device=device).masked_fill(input_sequence == 0, 1).bool()
