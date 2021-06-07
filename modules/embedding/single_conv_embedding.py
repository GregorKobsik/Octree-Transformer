import torch.nn as nn


class SingleConvolutionalEmbedding(nn.Module):
    def __init__(self, embed_dim, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(SingleConvolutionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
