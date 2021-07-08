import torch
import torch.nn as nn

from masks import look_ahead_mask, full_mask


class EncoderMultiDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_positions, token_embedding, generative_head, **_):
        """ Creates an instance of an encoder multi decoder transformer.

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
        super(EncoderMultiDecoder, self).__init__()

        self.embed_dim = embed_dim  # E
        self.num_positions = num_positions
        num_decoders = len(generative_head)

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
        self.transformer_decoder = nn.ModuleList(
            [
                nn.TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=num_layers,
                    norm=nn.LayerNorm(embed_dim),
                ) for _ in range(num_decoders)
            ]
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
        src = self.embedding[0].source(value, depth, pos)  # [N, S, E]

        # create padding mask
        padding_mask = self.embedding[0].src_padding_mask(value, depth)  # [N, S]

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

    def decode(self, value, depth, pos, memory, decoder_idx, final):
        """ Performs computations in the decoder part of the transformer.

        It embeds the target token sequence into the embedding space of the decoder and creates an upper triangular
        mask to allow only for autoregressive token access.

        Args:
            value: Target value token sequence - [N, T].
            depth: Target depth token sequence - [N, T].
            pos: Target position token sequences with a single token for each spatial dimension of the data - [N, T, A].
            memory: The output of the last encoder/decoder layer - [N, S/T, E].
            decoder_idx: Defines which decoder instance should be used.
            final: Defines if the current input is final, e.g. if the decoder should be 'autoregressive'.

        Return:
            The output of the last layer of the decoder in latent decoder space - [N, T, E].
        """
        # compute the embedding vector sequence for decoder input
        tgt = self.embedding[decoder_idx + 1].target(value, depth, pos)  # [N, T, E]
        tgt = self._prepend_sos_token(tgt)  # [N, T, E]

        # create autoregressive attention and padding masks
        tgt_len = tgt.shape[1]
        if final:
            attn_mask = look_ahead_mask(tgt_len, device=tgt.device)  # [T, T]
        else:
            attn_mask = full_mask(tgt_len, device=tgt.device)  # [T, T]
        padding_mask = self.embedding[decoder_idx + 1].tgt_padding_mask(value, depth)  # [N, T]

        # limit sequence length to max `num_position`
        tgt = tgt[:, :self.num_positions]  # [N, T, E]
        attn_mask = attn_mask[:self.num_positions, :self.num_positions]  # [T, T]
        padding_mask = padding_mask[:, :self.num_positions]  # [N, T]

        # decoder part of the transformer - pytorch expects, the sequence dimension to be first.
        out = self.transformer_decoder[decoder_idx](
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
        seq_enc, seq_dec = sequence

        # process encoder
        memory = self.encode(*seq_enc)  # [N, S, E]

        # process every other layer separatly
        for idx, seq_layer in enumerate(seq_dec):
            is_final = idx == len(seq_dec) - 1

            # process decoder
            z = self.decode(*seq_layer, memory, idx, is_final)  # [N, T*, E]

            # return logits in final layer
            if is_final:
                return self.head[idx](z, *seq_dec[-1])  # [N, T, V]