from random import randint

from .collate_utils import (
    get_min_batch_depth,
    pad_batch,
)


class EncoderDecoderCollate():
    """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'. """
    def __call__(self, batch):
        """ Pads and packs a list of samples for the 'encoder_decoder' architecture. """
        # get get the maximal usable depth value for every sample
        max_depth = get_min_batch_depth(batch)

        # select a random depth limit for this batch
        lim_depth = randint(2, max_depth)

        # extract layers as input for the encoder
        batch_src = [(v[d < lim_depth], d[d < lim_depth], p[d < lim_depth]) for v, d, p in batch]

        # extract last layer as input for the decoder & target
        batch_tgt = [(v[d == lim_depth], d[d == lim_depth], p[d == lim_depth]) for v, d, p in batch]

        # pad sequences
        src = pad_batch(batch_src)
        tgt = pad_batch(batch_tgt)

        # return as ((input_enc, input_dec), target)
        return (src, tgt), tgt
