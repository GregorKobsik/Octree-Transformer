import torch

SPATIAL_DIM = 3


def ancestor_mask(sequence, device=None):
    """ Creates a sparse ancestor mask, which allows to assign only to direct precedestors in the current block
        as well as to access the parent position. All other positions all masked out, no look-ahead possible.

        args:
            sequence: Input sequence with shape [S, N].
            device: Device on which the tensor will be created.

        return:
            Attention mask with shape [N, S, S].
    """
    seq_len, batch_size = sequence.shape  # [S, N]
    block_len = 2**SPATIAL_DIM
    num_blocks = seq_len // block_len

    # init attention mask: [N, S(input), S(target)]
    attn_mask = torch.full((batch_size, seq_len, seq_len), 0, device=device)
    attn_mask[:, 0, 0] = 1  # sos token
    # attn_mask[:, 1, 1:block_len] = 1  # initial token

    for n in range(batch_size):
        # create blocks - sparse fixed pattern
        for i in range(num_blocks):
            attn_mask[n, 0:1 + (i + 1) * block_len, 2 + i * block_len:2 + (i + 1) * block_len - 1] = 1
        attn_mask[n] = torch.tril(attn_mask[n])

        # mark parent - ancestor pattern
        ii = 0
        for i in range(seq_len):
            if sequence[i, n] == 2:
                attn_mask[n, 1 + block_len * ii:1 + block_len * (ii + 1), i + 1] = 1
                ii += 1

    # convert values to additive matrix with 0 or -Inf
    return torch.where(attn_mask == 1, 0.0, -float("Inf"))
