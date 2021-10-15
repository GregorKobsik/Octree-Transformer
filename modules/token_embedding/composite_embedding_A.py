import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .basic_embedding_A import BasicEmbeddingA
from .convolution_embedding_A import ConvolutionEmbeddingA
from .substitution_embedding import SubstitutionEmbedding
from .double_substitution_embedding import DoubleSubstitutionEmbedding


class CompositeEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim, **_):
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

        modules = []
        if resolution >= 2:
            modules += [BasicEmbeddingA(**kwargs)]
        if resolution >= 4:
            modules += [BasicEmbeddingA(**kwargs)]
        if resolution >= 8:
            modules += [BasicEmbeddingA(**kwargs)]
        if resolution >= 16:
            modules += [ConvolutionEmbeddingA(**kwargs, conv_size=2**(spatial_dim - 1))]
        if resolution >= 32:
            modules += [ConvolutionEmbeddingA(**kwargs, conv_size=2**spatial_dim)]
        if resolution >= 64:
            modules += [SubstitutionEmbedding(**kwargs, conv_size=2**spatial_dim)]
        if resolution >= 128:
            modules += [DoubleSubstitutionEmbedding(**kwargs, conv_size=2**spatial_dim)]
        if resolution >= 256:
            modules += [DoubleSubstitutionEmbedding(**kwargs, conv_size=2**spatial_dim)]

        # embeddings
        self.embeddings = nn.ModuleList(modules)

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

                # filter layers for embeddings
                if layer_depth < 6:  # only last layer
                    val_seq = val[dep == layer_depth]
                    dep_seq = dep[dep == layer_depth]
                    pos_seq = pos[dep == layer_depth]
                elif layer_depth == 6:  # penultimate and last layer
                    val_seq = torch.cat([val[dep == (layer_depth - 1)], val[dep == layer_depth]])
                    dep_seq = torch.cat([dep[dep == (layer_depth - 1)], dep[dep == layer_depth]])
                    pos_seq = torch.cat([pos[dep == (layer_depth - 1)], pos[dep == layer_depth]])
                elif layer_depth in (7, 8):  # third-, second- and last layer
                    val_seq = torch.cat(
                        [val[dep == (layer_depth - 2)], val[dep == (layer_depth - 1)], val[dep == layer_depth]]
                    )
                    dep_seq = torch.cat(
                        [dep[dep == (layer_depth - 2)], dep[dep == (layer_depth - 1)], dep[dep == layer_depth]]
                    )
                    pos_seq = torch.cat(
                        [pos[dep == (layer_depth - 2)], pos[dep == (layer_depth - 1)], pos[dep == layer_depth]]
                    )

                # compute layer embedding
                layer_emb = embedding(
                    val_seq.unsqueeze(0),
                    dep_seq.unsqueeze(0),
                    pos_seq.unsqueeze(0),
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
