import torch
import torch.nn as nn

from masks import look_ahead_mask


class EncoderOnly(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_positions, token_embedding, generative_head, **_):
        """ Creates an instance of an encoder only transformer..

        It accepts different implementations of `token_embedding`s and `generative_head`s. The following abbrevations
        are used to reference the size and the content of a dimension in used tensors.

        Shapes:
            N: batch size
            S: source sequence length
            E: embedding dimension
            A: spatial dimension
            V: vocabulary size

        Args:
            embed_dim: Number of embedding dimensions used by the attention.
            num_heads: Number of heads used by the attention.
            num_layers: Number of encoder layers in the transformer.
            num_positions: Maximal length of processed input tokens. You can pass longer sequences as input, but they
                will be truncated before feeding into the transformer, but after the embedding. Thus longer sequences
                can be accepted by a non-basic embedding and possibly compressed to stay within the limit.
            token_embedding: Instance of an embedding layer, which embedds given sequences of tokens into an embedding
                space, which is the direct input for the transformer layers.
            generative_head: Instance of a head layer, which transforms the output of the transformer into logits.

        """
        super(EncoderOnly, self).__init__()

        self.embed_dim = embed_dim  # E
        self.num_positions = num_positions

        # token embedding
        self.embedding = token_embedding

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # generative head
        self.head = generative_head

    def _prepend_sos_token(self, x):
        """ Shifts given sequence one token to right and fills missing token with start of sequence token. """
        batch_size = x.shape[0]
        sos = torch.ones(batch_size, 1, self.embed_dim, device=x.device) * self.sos  # [N, 1, E]
        return torch.cat([sos, x[:, :-1]], axis=1)  # [N, S, E]

    def _transpose(self, x):
        """ Transposes the first and second dimension of the input tensor. """
        return torch.transpose(x, 0, 1)

    def encode(self, value, depth, pos):
        """ Performs computations in the encoder part of the transformer.

        It embedds the given token sequences into the embedding space, given the `token_embedding`. Next, it creates
        a upper triangular mask. Therefore the transformer can access its input tokens only autoregressiv.

        Args:
            value: Value token sequence - [N, S].
            depth: Depth token sequence - [N, S].
            pos: Position token sequences with a single token for each spatial dimension of the data - [N, S, A].

        Return:
            The output of the last layer of the encoder in latent encoder space - [N, S, E].
        """

        # compute the embedding vector sequence for encoder input
        src = self.embedding(value, depth, pos)  # [N, S, E]
        src = self._prepend_sos_token(src)  # [N, S, E]

        # create (optional: autoregressive attention and) padding mask
        src_len = src.shape[1]
        attn_mask = look_ahead_mask(src_len, device=src.device)  # [S, S]
        padding_mask = self.embedding.padding_mask(value, depth, pos)  # [N, S]

        # limit sequence length to max `num_position`
        src = src[:, :self.num_positions]  # [N, S, E]
        attn_mask = attn_mask[:self.num_positions, :self.num_positions]  # [S, S]
        padding_mask = padding_mask[:, :self.num_positions]  # [N, S]

        # encoder part of the transformer - pytorch expects, the sequence dimension to be first.
        out = self.transformer_encoder(
            src=self._transpose(src),  # [S, N, E]
            mask=attn_mask,  # Optional: [S, S] or None
            src_key_padding_mask=padding_mask,  # [N, S]
        )  # [S, N, E]
        return self._transpose(out)  # [N, S, E]

    def forward(self, sequence):
        """ Performs a full transformer pass of the input sequence through embedding, transformer and generative head.

        Args:
            sequence: Tuple containing input sequences as (value, depth, position) sequences with the shapes ([N, S],
            [N, S], [N, S, A]).

        Return:
            Logits which describe the autoregressive likelihood of the next target token with shape [N, S, V].
        """
        # execute one transformer step on the input sequences
        z = self.encode(*sequence)  # [N, S, E]
        # return logits
        return self.head(z, *sequence)  # [N, S, V]
