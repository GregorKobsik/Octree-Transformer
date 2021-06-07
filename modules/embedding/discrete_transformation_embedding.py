import torch.nn as nn

from modules.embedding import BasicEmbedding


class DiscreteTransformationEmbedding(BasicEmbedding):
    def __init__(self, num_vocab, embed_dim, tree_depth, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            tree_depth: Number of depth layers used for  sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(DiscreteTransformationEmbedding, self).__init__(num_vocab, embed_dim, tree_depth, spatial_dim)

        # embeddings - modify target embedding, as tokens of last layer are summarized in trinary encoding
        self.tgt_value_embedding = nn.Embedding(num_vocab**2**spatial_dim + 1, embed_dim, padding_idx=0)
