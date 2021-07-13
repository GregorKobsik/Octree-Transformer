import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .basic_embedding_A import BasicEmbeddingA
from .half_conv_embedding_A import HalfConvolutionalEmbeddingA
from .single_conv_embedding_A import SingleConvolutionalEmbeddingA
# from .substitution_embedding import SubstitutionEmbedding


class CompositeEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a different embedding for each depth layer, possibly reducing the overall sequence lenght.
        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(CompositeEmbeddingA, self).__init__()

        kwargs = {
            "num_vocab": num_vocab,
            "embed_dim": embed_dim,
            "resolution": resolution,
            "spatial_dim": spatial_dim,
        }

        # embeddings
        self.embeddings = nn.ModuleList(
            [
                BasicEmbeddingA(**kwargs),
                BasicEmbeddingA(**kwargs),
                BasicEmbeddingA(**kwargs),
                HalfConvolutionalEmbeddingA(**kwargs),
                SingleConvolutionalEmbeddingA(**kwargs),
                # SubstitutionEmbedding(**kwargs),
            ]
        )

    def forward(self, value, depth, position):
        """ Transform sequences of token into an embedding space.

        Args:
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Token sequence in the embedding space.
        """
        batch_depth = torch.max(depth)
        batch_size = len(value)

        x = []
        padding_mask = []

        # process each sample individually
        for i in range(batch_size):
            # extract value, depth and position sequence of current sample
            val, dep, pos = value[i], depth[i], position[i]
            emb = torch.tensor([], device=value.device)

            # embed layerwise
            for layer_idx, embedding in enumerate(self.embeddings):
                layer_depth = layer_idx + 1
                if layer_depth > batch_depth:
                    break  # reached max depth layer

                # TODO: add penult layer for substitution!
                # compute layer embedding
                layer_emb = embedding(
                    val[dep == layer_depth].unsqueeze(0),
                    dep[dep == layer_depth].unsqueeze(0),
                    pos[dep == layer_depth].unsqueeze(0),
                )[0]
                emb = torch.cat([emb, layer_emb])

            # append embedding
            x += [emb]
            padding_mask += [torch.zeros(emb.shape[0], dtype=torch.bool, device=value.device)]

        # create padding mask
        self.mask = pad_sequence(padding_mask, batch_first=True, padding_value=1)
        # pad embedding sequence
        return pad_sequence(x, batch_first=True, padding_value=0.0)

    def padding_mask(self, value, depth, position):
        """ Creates a token padding mask based on sequence tokens.

        Note: Creates the padding mask during the forward pass. Only a getter function.

        Args:
            value: unused.
            depth: unused.
            position: unused.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return self.mask
