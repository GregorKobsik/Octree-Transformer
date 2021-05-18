import torch


def padding_mask(input_sequence, device=None):
    """ Create a padding mask for the given input.

        Always assumens '0' as a padding value.
        input_sequence has the shape (S, N)

        PyTorch Transformer defines 'src_key_padding_mask' with shape (N, S),
        where the input shape of 'src' is (S, N, E).
        Therefor we need to transpose the dimensions of the created mask.
        """
    mask = torch.zeros_like(input_sequence, device=device).masked_fill(input_sequence == 0, 1).bool()  # [S, N]
    return mask.transpose(0, 1)  # [N, S]
