import torch
import torch.nn as nn

from masks import look_ahead_mask


class EncoderDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_positions, token_embedding, generative_head, **_):
        """ Creates an instance of an encoder decoder transformer.

        It accepts different implementations of `token_embedding`s and `generative_head`s. The following abbrevations
        are used to reference the size and the content of a dimension in used tensors.

        Shapes:
            N: batch size
            S: source sequence length
            T: target sequence length
            E: embedding dimension
            A: spatial dimension
            V: vocabulary size

        Args:
            embed_dim: Number of embedding dimensions used by the attention.
            num_heads: Number of heads used by the attention.
            num_layers: Number of layers for each the 'decoder' and 'encoder' part of the transformer.
            num_positions: Maximal length of processed input tokens. You can pass longer sequences as input, but they
                will be truncated before feeding into the transformer, but after the embedding. Thus longer sequences
                can be accepted by a non-basic embedding and possibly compressed to stay within the limit.
            token_embedding: Instance of an embedding layer, which embedds given sequences of tokens into an embedding
                space, which is the direct input for the transformer layers.
            generative_head: Instance of a head layer, which transforms the output of the transformer into logits.

        """
        super(EncoderDecoder, self).__init__()

        self.embed_dim = embed_dim  # E
        self.num_positions = num_positions

        # token embedding
        if isinstance(token_embedding, nn.ModuleList):
            self.enc_embedding = token_embedding[0]
            self.dec_embedding = token_embedding[1]
        else:
            self.enc_embedding = token_embedding
            self.dec_embedding = token_embedding

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
        a full mask therefore every token can access every other token.

        Args:
            value: Value token sequence - [N, S].
            depth: Depth token sequence - [N, S].
            pos: Position token sequences with a single token for each spatial dimension of the data - [N, S, A].

        Return:
            The output of the last layer of the encoder in latent encoder space - [N, S, E].
        """

        # compute the embedding vector sequence for encoder input
        src = self.enc_embedding(value, depth, pos)  # [N, S, E]

        # create padding mask
        padding_mask = self.enc_embedding.padding_mask(value, depth, pos)  # [N, S]

        # limit sequence length to max `num_position`
        src = src[:, :self.num_positions]  # [N, S, E]
        padding_mask = padding_mask[:, :self.num_positions]  # [N, S]

        # encoder part of the transformer - pytorch expects, the sequence dimension to be first.
        out = self.transformer_encoder(
            src=self._transpose(src),  # [S, N, E]
            mask=None,
            src_key_padding_mask=padding_mask,  # [N, S]
        )  # [S, N, E]
        return self._transpose(out)  # [N, S, E]

    def decode(self, value, depth, pos, memory):
        """ Performs computations in the decoder part of the transformer.

        It embeds the target token sequence into the embedding space of the decoder and creates an upper triangular
        mask to allow only for autoregressive token access.

        Args:
            value: Target value token sequence - [N, T].
            depth: Target depth token sequence - [N, T].
            pos: Target position token sequences with a single token for each spatial dimension of the data - [N, T, A].
            memory: The output of the last encoder layer - [N, S, E].

        Return:
            The output of the last layer of the decoder in latent decoder space - [N, T, E].
        """
        # compute the embedding vector sequence for decoder input
        tgt = self.dec_embedding(value, depth, pos)  # [N, T, E]
        tgt = self._prepend_sos_token(tgt)  # [N, T, E]

        # create autoregressive attention and padding masks
        tgt_len = tgt.shape[1]
        attn_mask = look_ahead_mask(tgt_len, device=tgt.device)  # [T, T]
        padding_mask = self.dec_embedding.padding_mask(value, depth, pos)  # [N, T]

        # limit sequence length to max `num_position`
        tgt = tgt[:, :self.num_positions]  # [N, T, E]
        attn_mask = attn_mask[:self.num_positions, :self.num_positions]  # [T, T]
        padding_mask = padding_mask[:, :self.num_positions]  # [N, T]

        # decoder part of the transformer - pytorch expects, the sequence dimension to be first.
        out = self.transformer_decoder(
            tgt=self._transpose(tgt),  # [T, N, E]
            memory=self._transpose(memory),  # [S, N, E]
            tgt_mask=attn_mask,  # [T, T]
            tgt_key_padding_mask=padding_mask  # [N, T]
        )  # [T, N, E]
        return self._transpose(out)  # [N, T, E]

    def forward(self, sequence):
        """ Performs a full transformer pass of the input sequence through embedding, transformer and generative head.

        Args:
            sequence: Tuple containing input sequences as a tuple of (encoder_sequence, decoder_sequence), where each
                of the elements is another tuple of (value, depth, position) sequence inputs for the encoder and decoder
                with the shape ([N, S/T], [N, S/T], [N, S/T, A]), respectively.

        Return:
            Logits which describe the autoregressive likelihood of the next target token, with shape [N, T, V].
        """
        # execute one transformer step on the input sequences
        seq_enc, seq_dec = sequence
        memory = self.encode(*seq_enc)  # [N, S, E]
        z = self.decode(*seq_dec, memory)  # [N, T, E]
        # return logits
        return self.head(z, *seq_dec)  # [N, T, V]
