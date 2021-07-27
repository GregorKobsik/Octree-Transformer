from random import randint

from .collate_utils import (
    get_min_batch_depth,
    pad_batch,
)


class EncoderDecoderCollate():
    def __init__(self, embeddings, resolution):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            embeddings: Defines the used token embeddings in the shape transformer.
        """
        self.embeddings = embeddings

    """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'. """

    def __call__(self, batch):
        """ Pads and packs a list of samples for the 'encoder_decoder' architecture. """
        # get get the maximal usable depth value for every sample
        max_depth = get_min_batch_depth(batch)

        # select a random depth limit for this batch
        lim_depth = randint(2, max_depth)

        # extract layers as input for the encoder
        batch_enc = [(v[d < lim_depth], d[d < lim_depth], p[d < lim_depth]) for v, d, p in batch]

        # extract last layer as input for the decoder & target
        if self.embeddings[1] in ('substitution'):
            # extract last layer as input for the decoder & target
            batch_dec = [
                (
                    v[(lim_depth - 1 <= d) & (d <= lim_depth)],
                    d[(lim_depth - 1 <= d) & (d <= lim_depth)],
                    p[(lim_depth - 1 <= d) & (d <= lim_depth)],
                ) for v, d, p in batch
            ]
            batch_tgt = [(v[d == lim_depth], d[d == lim_depth], p[d == lim_depth]) for v, d, p in batch]
        elif self.embeddings[1] in ('double_substitution'):
            # extract third-, second-, and last layer as input for the decoder & target
            batch_dec = [
                (
                    v[(lim_depth - 2 <= d) & (d <= lim_depth)],
                    d[(lim_depth - 2 <= d) & (d <= lim_depth)],
                    p[(lim_depth - 2 <= d) & (d <= lim_depth)],
                ) for v, d, p in batch
            ]
            batch_tgt = [(v[d == lim_depth], d[d == lim_depth], p[d == lim_depth]) for v, d, p in batch]
        else:
            # extract last layer as input for the decoder & target
            batch_dec = [(v[d == lim_depth], d[d == lim_depth], p[d == lim_depth]) for v, d, p in batch]
            batch_tgt = batch_dec

        # pad sequences
        enc = pad_batch(batch_enc)
        dec = pad_batch(batch_dec)
        tgt = pad_batch(batch_tgt)

        # return as ((input_enc, input_dec), target)
        return (enc, dec), tgt
