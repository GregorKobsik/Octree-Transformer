import torch
from torch.nn.utils.rnn import pad_sequence


def to_sequences(batch):
    """ Transform a list on numpy arrays into sequences of pytorch tensors. """
    batch = [(torch.tensor(v), torch.tensor(d), torch.tensor(p)) for v, d, p in batch]

    # unpack batched sequences
    return zip(*batch)


def pad_batch(batch):
    """ Unpack batch and pad each sequence to a tensor of equal length. """
    val, dep, pos = to_sequences(batch)

    # pad each sequence
    val_pad = pad_sequence(val, batch_first=True, padding_value=0)
    dep_pad = pad_sequence(dep, batch_first=True, padding_value=0)
    pos_pad = pad_sequence(pos, batch_first=True, padding_value=0)

    return val_pad, dep_pad, pos_pad


def get_min_batch_depth(batch):
    """ Compute the smallest max depth layer of all samples in the batch. """
    max_depth = float('Inf')
    for b in batch:
        b_depth = max(b[1])
        max_depth = b_depth if b_depth < max_depth else max_depth
    return max_depth
