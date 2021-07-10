from random import randint

from .collate_utils import (
    get_min_batch_depth,
    pad_batch,
)


class EncoderMultiDecoderCollate():
    def __init__(self, num_concat_layers):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            num_concat_layers: Defines number of layers which will be concatinated for the encoder.
        """
        self.num_concat_layers = num_concat_layers

    def __call__(self, batch):
        """ Packs a list of samples for the 'encoder_multi_decoder' architecture. """
        max_depth = get_min_batch_depth(batch)
        num_concat = self.num_concat_layers

        # concat the first `num_concat` layers
        batch_layer = [(v[d <= num_concat], d[d <= num_concat], p[d <= num_concat]) for v, d, p in batch]
        seq = [pad_batch(batch_layer)]

        # select a random depth limit for this batch
        lim_depth = randint(num_concat, max_depth)

        # extract further layers separatly up to the depth limit
        for i in range(num_concat + 1, lim_depth + 1):
            batch_layer = [(v[d == i], d[d == i], p[d == i]) for v, d, p in batch]
            seq += [pad_batch(batch_layer)]

        # return as (sequence, target)
        return seq, seq[-1]
