import torch
import torch.nn as nn

from utils.masks import look_ahead_mask, full_mask


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        num_decoders,
        token_embedding,
        generative_head,
        dropout,
        num_classes,
        **_,
    ):
        """ Creates an instance of a transformer module.

        It accepts different implementations of `token_embedding`s and `generative_head`s, which define the architecture
        of the transformer. It can be either an 'encoder_only', 'encoder_decoder' or an 'encoder_multi_decoder'.

        The following abbrevations are used to reference the size and the content of a dimension in used tensors.

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
            num_decoders: Defines the number of decoder instances.
            token_embedding: Instance of an embedding layer, which embedds given sequences of tokens into an embedding
                space, which is the direct input for the transformer layers.
            generative_head: Instance of a head layer, which transforms the output of the transformer into logits.
            dropout: The dropout value.
            num_classes: If bigger, that one the transformer will be class conditional
        """
        super(Transformer, self).__init__()

        self.embed_dim = embed_dim  # E

        # token embedding
        self.embedding = token_embedding

        # start of sequence token
        if num_classes > 1:
            self.cls_embedding = nn.Embedding(num_classes, embed_dim)
        else:
            self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(self.sos)
        self.cls_conditional = num_classes > 1

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
        )

        # transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
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

    def process(self, seq, memory, padding_mask, layer_idx, is_final, cls):
        """ Performs computations in the decoder part of the transformer.

        It embeds the target token sequence into the embedding space of the decoder and creates an upper triangular
        mask to allow only for autoregressive token access.

        Args:
            seq: Token layer sequence in embedding space - [N, L, E]
            memory: Output of the last transformer layer - [N, M, E].
            padding_mask: Value token layer sequence padding mask - [N, L].
            layer_idx: Defines which transformer layer should be used.
            is_final: Defines if the current layer is final, e.g. if the transformer should be 'autoregressive'.
            cls: class label for conditional generation.

        Return:
            The output of the last layer of the decoder in latent decoder space - [N, L, E].
        """
        # create attention mask
        seq_len = seq.shape[1]
        if is_final:  # attention mask is autoregressive in the final layer
            attn_mask = look_ahead_mask(seq_len, device=seq.device)  # [L, L]
            # shift sequence by one token to right to predict tokens autoregressively
            if self.cls_conditional:
                seq = torch.cat([self.cls_embedding(cls).unsqueeze(1), seq[:, :-1]], dim=1)
            else:
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

    def forward(self, sequence, cls):
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
            if idx < seq_len - 1:  # intermediate layer
                memory = self.compute_memory(seq_layer, memory, idx, False, cls)  # [N, L, E]
            else:  # only final layer
                return self.compute_logits(seq_layer, memory, idx, cls)  # [N, T, V]

    def compute_memory(self, seq_layer, memory, idx, is_final, cls):
        """ Computes the output of the corresponding transformer layer, without processing the corresponding head.

        Args:
            seq_layer: Token sequence of a single layer as a tuple of (value, depth, position), with the shapes
                ([N, L], [N, L], [N, L, A]).
            memory: Memory sequence of the previous transformer layer, with the shape [N, T]. Should be 'None' iff
                `idx` is 0.
            idx: Index of the transformer layer.
            is_final: Defines the used mask. True - uses an autoregressive mask, False - each token can access each
                other token.
            cls: class label for conditional generation.

        Return:
            Memory latent vector of the selecter transformer layer with the shape [N, L, E].
        """
        # embed sequence tokens
        emb = self.embedding[idx](*seq_layer)  # [N, L, E]
        seq_mask = self.embedding[idx].padding_mask(*seq_layer)  # [N, L]

        # compute memory / process sequence
        return self.process(emb, memory, seq_mask, idx, is_final, cls)  # [N, L, E]

    def compute_logits(self, seq_layer, memory, idx, cls):
        """ Performs a full pass of a single transformer layer to computes the logits of given sequence.

        Each token can access previous tokens in `seq_layer` only autoregressivelly. All tokens of the `memory`
            sequence can be accessed be each token of `seq_layer`.

        Args:
            seq_layer: Token sequence of a single layer as a tuple of (value, depth, position), with the shapes
                ([N, L], [N, L], [N, L, A]).
            memory: Memory sequence of the previous transformer layer, with the shape [N, T]. Should be 'None' iff
                `idx` is 0.
            idx: Index of the transformer layer.
            cls: class label for conditional generation.

        Return
            Logits of the given layer token sequence with the shape [N, L, V]
        """
        # compute memory
        memory = self.compute_memory(seq_layer, memory, idx, True, cls)  # [N, L, E]

        # return logits
        return self.head[idx](memory, *seq_layer)  # [N, T, V]
