import math
from random import randint

from .collate_utils import (
    get_min_batch_depth,
    pad_batch,
)


class EncoderMultiDecoderCollate():
    def __init__(self, embeddings, resolution):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            embeddings: Defines the used token embeddings in the shape transformer.
            resolution: Maximum side length of input data.
        """
        self.embeddings = embeddings
        # Defines number of layers which will be concatinated for the encoder.
        self.num_concat_layers = 1 + int(math.log2(resolution)) - len(embeddings)

    def __call__(self, batch):
        """ Packs a list of samples for the 'encoder_multi_decoder' architecture.

            Extract each layer of value, depth and position individually as layer sequence for each sample. Concat the
            first `num_concat_layers`. For 'substitution' embedding prepend additionally the previous layer to the
            current layer. Use the last layer sequence as target. For 'substitution' embedding, use only the current
            layer as target.
        """
        # select a random depth limit for this batch
        max_depth = get_min_batch_depth(batch)
        limit = randint(0, max_depth - self.num_concat_layers)

        seq = []
        for embedding_idx, embedding in enumerate(self.embeddings):
            # select the lower (lo) and upper (up) layer depth bounds for the current embedding
            if embedding_idx == 0:
                # concat the first `num_concat` layers for the encoder
                lo = 1
                up = self.num_concat_layers
            elif embedding in ('substitution'):
                # select previous and last layer for 'substitution' embedding
                lo = embedding_idx + self.num_concat_layers - 1
                up = embedding_idx + self.num_concat_layers
            elif embedding in ('double_substitution'):
                # select previous and last layer for 'substitution' embedding
                lo = embedding_idx + self.num_concat_layers - 2
                up = embedding_idx + self.num_concat_layers
            else:
                # get only a single depth layer
                lo = embedding_idx + self.num_concat_layers
                up = lo

            # extract value, depth and position sequences for each sample in batch
            batch_layer = [
                (v[(lo <= d) & (d <= up)], d[(lo <= d) & (d <= up)], p[(lo <= d) & (d <= up)]) for v, d, p in batch
            ]
            seq += [batch_layer]

            # filter sequence for target value, depth and position
            if embedding in ('substitution', 'double_substitution'):
                tgt = [(v[d == up], d[d == up], p[d == up]) for v, d, p in batch_layer]
            else:
                tgt = batch_layer

            if embedding_idx >= limit:
                break  # reached embedding/layer depth limit

        # pad each sequence layer
        seq = [pad_batch(batch_layer) for batch_layer in seq]
        tgt = pad_batch(tgt)

        # return as (sequence, target)
        return seq, tgt
