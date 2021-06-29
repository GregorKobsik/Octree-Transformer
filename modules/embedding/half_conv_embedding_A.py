import math
import torch.nn as nn


class HalfConvolutionalEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^(`spatial_dim` - 1).

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(HalfConvolutionalEmbeddingA, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # convolutions
        s = 2**(spatial_dim - 1)
        self.convolution = nn.Conv1d(embed_dim, embed_dim, kernel_size=s, stride=s)

    def source(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder and reduce number of tokens.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into higher dimension
        x = self.value_embedding(value)  # [N, S, E]
        x = x + self.depth_embedding(depth)  # [N, S, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, E]

        # convolute tokens to reduce sequence length
        return self.convolution(x.transpose(1, 2)).transpose(1, 2)  # [N, S', E]
