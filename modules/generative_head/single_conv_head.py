import torch.nn as nn


class SingleConvolutionalHead(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a concolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.

        TODO: implement this module!
        """
        super(SingleConvolutionalHead, self).__init__()
        print("ERROR: Not implemented, yet")
        raise ValueError
