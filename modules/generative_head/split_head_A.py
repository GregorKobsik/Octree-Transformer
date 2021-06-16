import torch
import torch.nn as nn


class SplitHeadA(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a linear transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(SplitHeadA, self).__init__()
        self.chunk_size = 2**spatial_dim
        e_dim = embed_dim // self.chunk_size

        self.linear = nn.Linear(e_dim, num_vocab + 1, bias=False)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Args:
            x: Output of the transformer, the latent vector [N, T, E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        batch_size, num_tokens, embed_dim = x.shape  # [N, T', E]

        # reshape tensor along embedding dimension to create multiple tokens
        x = torch.reshape(x, (batch_size, num_tokens * self.chunk_size, -1))  # [N, T, E']

        # compute logits for each token
        return self.linear(x)  # [N, T, V]
