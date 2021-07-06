from random import randint

from data.collate import AbstractCollate


class PadCollate(AbstractCollate):
    def __init__(self, architecture):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            architecture: Defines the underlying transformer architecture.
        """
        super(PadCollate, self).__init__(architecture)

    def _get_max_batch_depth(self, batch):
        """ Compute the smallest max depth layer of all samples in the batch. """
        max_depth = float('Inf')
        for b in batch:
            b_depth = max(b[1])
            max_depth = b_depth if b_depth < max_depth else max_depth
        return max_depth

    def encoder_only(self, batch):
        """ Pads and packs a list of samples for the 'encoder_only' architecture. """
        # pad batched sequences with '0' to same length
        seq_pad = self._pad_batch(batch)

        # return as (sequence, target)
        return seq_pad, seq_pad

    def encoder_decoder(self, batch):
        """ Pads and packs a list of samples for the 'encoder_decoder' architecture. """
        # get maximal depth value
        max_depth = self._get_max_batch_depth(batch)

        # select a random depth limit for this batch
        lim_depth = randint(2, max_depth)

        # extract layers as input for the encoder
        batch_src = [(v[d < lim_depth], d[d < lim_depth], p[d < lim_depth]) for v, d, p in batch]

        # extract last layer as input for the decoder & target
        batch_tgt = [(v[d == lim_depth], d[d == lim_depth], p[d == lim_depth]) for v, d, p in batch]

        # pad sequences
        src_pad = self._pad_batch(batch_src)
        tgt_pad = self._pad_batch(batch_tgt)

        # return as ((input_enc, input_dec), target)
        return (src_pad, tgt_pad), tgt_pad

    def autoencoder(self, batch):
        """ Pads and packs a list of samples for the 'autoencoder' architecture. """
        # get maximal depth value
        max_depth = self._get_max_batch_depth(batch)

        # extract last layer
        batch = [(v[d == max_depth], d[d == max_depth], p[d == max_depth]) for v, d, p in batch]

        # pad sequences and return tensors
        seq_pad = self._pad_batch(batch)

        # return as (sequence, target)
        return seq_pad, seq_pad
