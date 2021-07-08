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
            L: layer sequence length
            M: memory length
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
        assert len(token_embedding) == len(generative_head), "Number of embeddings and heads is not equal."
        num_decoders = len(generative_head) - 1

        # token embedding
        self.embedding = token_embedding

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )

        # transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )

        # encoder multi decoder transformer
        self.emd_transformer = nn.ModuleList()
        self.emd_transformer.append(
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(embed_dim),
            )
        )
        self.emd_transformer.extend(
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
        """ Shifts given sequence one token to right and pads with start of sequence (sos) token. """
        batch_size = x.shape[0]
        sos = torch.ones(batch_size, 1, self.embed_dim, device=x.device) * self.sos  # [N, 1, E]
        return torch.cat([sos, x[:, :-1]], axis=1)  # [N, S, E]

    def _transpose(self, x):
        """ Transposes the first and second dimension of the input tensor. """
        return torch.transpose(x, 0, 1)

    def process(self, seq, memory, padding_mask, layer_idx, is_final):
        """ Performs computations in the decoder part of the transformer.

        It embeds the target token sequence into the embedding space of the decoder and creates an upper triangular
        mask to allow only for autoregressive token access.

        Args:
            seq: Token layer sequence in embedding space - [N, L, E]
            memory: Output of the last transformer layer - [N, M, E].
            padding_mask: Value token layer sequence padding mask - [N, L].
            layer_idx: Defines which transformer layer should be used.
            is_final: Defines if the current layer is final, e.g. if the transformer should be 'autoregressive'.

        Return:
            The output of the last layer of the decoder in latent decoder space - [N, L, E].
        """
        # limit sequence length to max `num_position`
        seq = seq[:, :self.num_positions]  # [N, L, E]
        padding_mask = padding_mask[:, :self.num_positions]  # [N, L]

        # create attention mask
        seq_len = seq.shape[1]
        if is_final:  # attention mask is autoregressive in the final layer
            attn_mask = look_ahead_mask(seq_len, device=seq.device)  # [L, L]
            # shift sequence by one token to right to predict tokens autoregressively
            seq = self._prepend_sos_token(seq)  # [N, L, E]
        else:  # otherwise we allow access to all tokens
            attn_mask = full_mask(seq_len, device=seq.device)  # [L, L]

        # process one transformer layer
        if layer_idx == 0:  # encoder part of the transformer
            out = self.emd_transformer[0](
                src=self._transpose(seq),  # [L, N, E], pytorch expects the sequence dimension to be first
                mask=attn_mask,  # [L, L]
                src_key_padding_mask=padding_mask,  # [N, L]
            )  # [S, N, E]
        else:  # decoder part of the transformer
            out = self.emd_transformer[layer_idx](
                tgt=self._transpose(seq),  # [L, N, E], pytorch expects the sequence dimension to be first
                memory=self._transpose(memory),  # [M, N, E]
                tgt_mask=attn_mask,  # [L, L]
                tgt_key_padding_mask=padding_mask  # [N, L]
            )  # [T, N, E]

        return self._transpose(out)  # [N, S/T, E]

    def forward(self, sequence):
        """ Performs a full transformer pass of the input sequence through embedding, transformer and generative head.

        Args:
            sequence: List containing input sequences, where each element is a tuple of (value, depth, position)
                sequence layer for the transformer with the shape ([N, L], [N, L], [N, L, A]), respectively.

        Return:
            Logits which describe the autoregressive likelihood of the next target token, with shape [N, T, V].
        """
        seq_len = len(sequence)
        memory = None

        # process sequence layers individually
        for idx, seq_layer in enumerate(sequence):
            is_final = idx == seq_len - 1

            # embed sequence tokens
            emb = self.embedding[idx](*seq_layer)  # [N, L, E]
            seq_mask = self.embedding[idx].padding_mask(*seq_layer)  # [N, L]

            # compute memory / process sequence
            memory = self.process(emb, memory, seq_mask, idx, is_final)  # [N, L, E]

            # return logits
            if is_final:  # compute only for final layer
                return self.head[idx](memory, *seq_layer)  # [N, T, V]
