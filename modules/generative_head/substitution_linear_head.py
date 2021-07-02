import torch
import torch.nn as nn


class SubstitutionLinearHead(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a concolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(SubstitutionLinearHead, self).__init__()
        self.stride = 2**spatial_dim
        self.conv_dim = embed_dim // self.stride

        self.linear_1 = nn.Linear(embed_dim, self.stride * self.conv_dim)
        self.linear_2 = nn.Linear(self.conv_dim, self.stride * self.conv_dim)
        self.linear_voc = nn.Linear(self.conv_dim, num_vocab + 1, bias=False)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Transforms one token of the latent vector into multiple tokens of the target vector through de-convolutional
        operations. In the case of a quadtree one token is responsible for up to 16 target tokens. In the case of a
        octree one token is responsible for up to 64 target tokens. Only tokens, which correspond to a mixed target
        value token in the penultimate layer are transformed into target sequence tokens.

        Args:
            x: Output of the transformer, the latent vector [N, T'', E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        batch_size = value.shape[0]

        # create intermediate tensor to hold values of penultimate layer
        penult_idx = torch.argmax(depth, dim=1)
        penult_val = torch.zeros((batch_size, torch.max(penult_idx)), device=x.device)  # [N, T'']

        # get values of penultimate layer - discard last layer of value tokens
        for i in range(batch_size):
            penult_val[i, :penult_idx[i]] = value[i, :penult_idx[i]]

        # create intermediate tensor to hold mixed tokens
        penult_len = torch.sum(penult_val == 2, dim=1)
        y = torch.zeros((batch_size, int(torch.max(penult_len)), self.conv_dim), device=x.device)  # [N, T', C]

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        x = self.linear_1(x).reshape(batch_size, -1, self.conv_dim)  # [N, T'', C]

        # select only latent vectors, which correspond to mixed tokens in the penultimate layer
        for i in range(batch_size):
            y[i, :penult_len[i]] = x[i, penult_val[i] == 2]  # [N, T', C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y = self.linear_2(y).reshape(batch_size, -1, self.conv_dim)  # [N, T, C]

        # compute logits of generated tokens
        return self.linear_voc(y)  # [N, T, V]
