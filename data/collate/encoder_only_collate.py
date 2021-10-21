import torch

from .collate_utils import pad_batch


class EncoderOnlyCollate():
    """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'. """

    def __call__(self, batch):
        """ Pads and packs a list of samples for the 'encoder_only' architecture. """
        # pad batched sequences with '0' to same length
        val, dep, pos, cls = pad_batch(batch)
        seq = [(val, dep, pos)]
        cls = torch.stack(cls)

        # return as (sequence, target)
        return seq, seq[-1], cls
