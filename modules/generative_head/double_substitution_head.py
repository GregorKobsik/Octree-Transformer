import torch
import torch.nn as nn

from ..utils import Deconvolution, BlockConvolution, Linear


class DoubleSubstitutionHead(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, spatial_dim, conv_size, **_):
        """ Performs a twice a substitution transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(DoubleSubstitutionHead, self).__init__()
        self.embed_dim = embed_dim

        # deconvolutions
        self.deconvolution_2 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.deconvolution_1 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.deconvolution_0 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.spatial_encoding = spatial_encoding

        # head
        self.linear = Linear(embed_dim, num_vocab)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Transforms one token of the latent vector into multiple tokens of the target vector through de-convolutional
        operations. In the case of a quadtree one token is responsible for up to 16 target tokens. In the case of a
        octree one token is responsible for up to 64 target tokens. Only tokens, which correspond to a mixed target
        value token in the penultimate layer are transformed into target sequence tokens.

        Args:
            x: Output of the transformer, the latent vector [N, T'', E].
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            pos: Position token sequence, with penultimate and last layer.

        Return
            Logits of target value sequence.
        """
        batch_size = value.shape[0]
        max_depth = torch.max(depth)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)
        len_2 = torch.sum(depth == (max_depth - 2), dim=1)

        # create intermediate list to hold values
        val_1 = torch.zeros((batch_size, torch.max(len_1)), device=value.device)
        val_2 = torch.zeros((batch_size, torch.max(len_2)), device=value.device)

        # splitt input in second-last (1) layer
        for i in range(batch_size):
            val_2[i, :len_2[i]] = value[i, :len_2[i]]
            val_1[i, :len_1[i]] = value[i, len_2[i]:len_2[i] + len_1[i]]

        # compute the number of mixed tokens in mask
        mix_1 = torch.sum(val_1 == 2, dim=1)
        mix_2 = torch.sum(val_2 == 2, dim=1)

        # create intermediate list to hold vectors
        x_0 = torch.zeros((batch_size, torch.max(mix_1), self.embed_dim), device=value.device)
        x_1 = torch.zeros((batch_size, torch.max(mix_2), self.embed_dim), device=value.device)

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_2 = self.deconvolution_2(x)
        # select only latent vectors, which correspond to mixed tokens in third-last layer
        for i in range(batch_size):
            mix_2_mask_i = (val_2[i] == 2)[:len(y_2[i])]  # handle overflow/clipped values in the embedding
            x_1[i, :torch.sum(mix_2_mask_i)] = y_2[i, mix_2_mask_i]  # [N, T', C]

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_1 = self.deconvolution_1(x_1)
        # select only latent vectors, which correspond to mixed tokens in third-last layer
        for i in range(batch_size):
            mix_1_mask_i = (val_1[i] == 2)[:len(y_1[i])]  # handle overflow/clipped values in the embedding
            x_0[i, :torch.sum(mix_1_mask_i)] = y_1[i, mix_1_mask_i]  # [N, T', C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y_0 = self.deconvolution_0(x_0)  # [N, T, C]

        # add spatial decoding if available
        if self.spatial_encoding is not None:
            len_last = torch.sum(depth == max_depth, dim=1)
            assert((depth[:, -len_last:] == max_depth).all())
            y_0 = y_0 + self.spatial_encoding(pos[:, -len_last:])

        # compute logits of generated tokens
        return self.linear(y_0)  # [N, T, V]


class DoubleSubstitutionHeadAutoRegressive(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, spatial_dim, conv_size, **_):
        """ Performs a twice a substitution transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(DoubleSubstitutionHeadAutoRegressive, self).__init__()
        self.embed_dim = embed_dim

        # deconvolutions
        self.deconvolution_2 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.deconvolution_1 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.deconvolution_0 = Deconvolution(embed_dim, embed_dim, conv_size)

        self.convolution_2 = BlockConvolution(embed_dim, embed_dim, conv_size)
        self.convolution_1 = BlockConvolution(embed_dim, embed_dim, conv_size)
        self.convolution_0 = BlockConvolution(embed_dim, embed_dim, conv_size)

        self.spatial_encoding = spatial_encoding

        # head
        self.linear = Linear(embed_dim, num_vocab)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Transforms one token of the latent vector into multiple tokens of the target vector through de-convolutional
        operations. In the case of a quadtree one token is responsible for up to 16 target tokens. In the case of a
        octree one token is responsible for up to 64 target tokens. Only tokens, which correspond to a mixed target
        value token in the penultimate layer are transformed into target sequence tokens.

        Args:
            x: Output of the transformer, the latent vector [N, T'', E].
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            pos: Position token sequence, with penultimate and last layer.

        Return
            Logits of target value sequence.
        """
        batch_size = value.shape[0]
        max_depth = torch.max(depth)
        len_0 = torch.sum(depth == (max_depth), dim=1)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)
        len_2 = torch.sum(depth == (max_depth - 2), dim=1)

        # create intermediate list to hold values
        val_1 = torch.zeros((batch_size, torch.max(len_1)), device=value.device)
        val_2 = torch.zeros((batch_size, torch.max(len_2)), device=value.device)

        # split input in second-last (1) layer
        for i in range(batch_size):
            val_2[i, :len_2[i]] = value[i, :len_2[i]]
            val_1[i, :len_1[i]] = value[i, len_2[i]:len_2[i] + len_1[i]]

        # compute the number of mixed tokens in mask
        mix_1 = torch.sum(val_1 == 2, dim=1)
        mix_2 = torch.sum(val_2 == 2, dim=1)

        assert ((depth[:, -len_0:] == max_depth).all())

        emb_0 = self.value_embedding(value[:, -len_0:])
        # add spatial decoding if available
        if self.spatial_encoding is not None:
            emb_0 = emb_0 + self.spatial_encoding(pos[:, -len_0:])
        emb_0 = self.convolution_0(emb_0)

        emb_1 = torch.zeros((batch_size, torch.max(len_1), self.embed_dim), dtype=torch.float, device=value.device)
        # substitute all mixed token embeddings of penultimate layer, with token embeddings of last layer
        emb_1[val_1 == 2] = emb_0[:, (self.conv_size - 1)::self.conv_size]  # [N, T1, C]
        emb_1 = self.convolution_1(emb_1)

        emb_2 = torch.zeros((batch_size, torch.max(len_2), self.embed_dim), dtype=torch.float, device=value.device)
        # substitute all mixed token embeddings of third to last layer, with token embeddings of penultimate layer
        emb_2[val_2 == 2] = emb_1[:, (self.conv_size - 1)::self.conv_size]  # [N, T1, C]
        emb_2 = self.convolution_2(emb_2)

        # create intermediate list to hold vectors
        x_0 = torch.zeros((batch_size, torch.max(mix_1), self.embed_dim), device=value.device)
        x_1 = torch.zeros((batch_size, torch.max(mix_2), self.embed_dim), device=value.device)

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_2 = self.deconvolution_2(x)
        y_2 = y_2 + emb_2[:, :y_2.shape[1]]
        # select only latent vectors, which correspond to mixed tokens in third-last layer
        for i in range(batch_size):
            mix_2_mask_i = (val_2[i] == 2)[:len(y_2[i])]  # handle overflow/clipped values in the embedding
            x_1[i, :torch.sum(mix_2_mask_i)] = y_2[i, mix_2_mask_i]  # [N, T', C]

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_1 = self.deconvolution_1(x_1)
        y_1 = y_1 + emb_1[:, :y_1.shape[1]]
        # select only latent vectors, which correspond to mixed tokens in third-last layer
        for i in range(batch_size):
            mix_1_mask_i = (val_1[i] == 2)[:len(y_1[i])]  # handle overflow/clipped values in the embedding
            x_0[i, :torch.sum(mix_1_mask_i)] = y_1[i, mix_1_mask_i]  # [N, T', C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y_0 = self.deconvolution_0(x_0)  # [N, T, C]
        y_0 = y_0 + emb_0[:, :y_0.shape[1]]  # [N, T, C]

        # add spatial decoding if available
        if self.spatial_encoding is not None:
            len_last = torch.sum(depth == max_depth, dim=1)
            assert((depth[:, -len_last:] == max_depth).all())
            y_0 = y_0 + self.spatial_encoding(pos[:, -len_last:])

        # compute logits of generated tokens
        return self.linear(y_0)  # [N, T, V]
