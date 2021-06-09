import torch.nn as nn


class SingleConvolutionalHeadC(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a convolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(SingleConvolutionalHeadC, self).__init__()
        s = 2**spatial_dim
        intermediate_dim = int(embed_dim / s)

        self.deconv = nn.ConvTranspose1d(embed_dim, intermediate_dim, kernel_size=s, stride=s)
        self.linear = nn.Linear(intermediate_dim, num_vocab + 1, bias=False)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Args:
            x: Output of the transformer, the latent vector [N, T', E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        # deconvolute the latent space - create new tokens
        x = self.deconv(x.transpose(1, 2)).transpose(1, 2)  # [N, T, E]

        # compute logits for each token
        return self.linear(x)  # [N, T, V]
