from data.collate.collate_utils import (
    get_min_batch_depth,
    pad_batch,
)


class AutoencoderCollate():
    """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'. """
    def __call__(self, batch):
        """ Pads and packs a list of samples for the 'autoencoder' architecture. """
        # get get the maximal usable depth value for every sample
        max_depth = get_min_batch_depth(batch)

        # extract last layer
        batch = [(v[d == max_depth], d[d == max_depth], p[d == max_depth]) for v, d, p in batch]

        # pad sequences and return tensors
        seq_pad = pad_batch(batch)

        # return as (sequence, target)
        return seq_pad, seq_pad
