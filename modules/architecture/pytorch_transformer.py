import torch
import torch.nn as nn

from utils.masks import look_ahead_mask


class PytorchTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        token_embedding,
        generative_head,
        dropout,
        num_classes,
        **_,
    ):
        """ Creates an instance of a transformer module.

        It accepts different implementations of `token_embedding`s and `generative_head`s, which define the architecture
        of the transformer.

        The following abbrevations are used to reference the size and the content of a dimension in used tensors.

        Shapes:
            N: batch size
            L: sequence length
            E: embedding dimension
            A: spatial dimension
            V: vocabulary size

        Args:
            embed_dim: Number of embedding dimensions used by the attention.
            num_heads: Number of heads used by the attention.
            num_layers: Number of layers for each the 'decoder' and 'encoder' part of the transformer.
            token_embedding: Instance of an embedding layer, which embedds given sequences of tokens into an embedding
                space, which is the direct input for the transformer layers.
            generative_head: Instance of a head layer, which transforms the output of the transformer into logits.
            dropout: The dropout value.
            num_classes: If bigger than one, the transformer will be class conditional
        """
        super(PytorchTransformer, self).__init__()

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

        # transformer stack
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # generative head
        self.head = generative_head

    def _prepend_sos_token(self, x, cls):
        """ Shifts given sequence one token to right and pads with start of sequence (sos) token. """
        if self.cls_conditional:
            return torch.cat([self.cls_embedding(cls).unsqueeze(1), x[:, :-1]], dim=1)
        else:
            # TODO: return torch.cat([self.sos.unsqueeze(1), x[:, :-1]], dim=1)
            batch_size = x.shape[0]
            sos = torch.ones(batch_size, 1, self.embed_dim, device=x.device) * self.sos  # [N, 1, E]
            return torch.cat([sos, x[:, :-1]], axis=1)  # [N, S, E]

    def _transpose(self, x):
        """ Transposes the first and second dimension of the input tensor. """
        return torch.transpose(x, 0, 1)

    def forward(self, sequence, cls):
        """ Performs a transformer forward pass of the sequence through embedding, transformer and generative head.

        Args:
            sequence: List containing input sequences, where each element is a tuple of (value, depth, position)
                sequence layer for the transformer with the shape ([N, L], [N, L], [N, L, A]), respectively.
            cls: class label, optional if `num_classes` <= 1.

        Return:
            Logits which describe the autoregressive likelihood of the next target token, with shape [N, T, V].
        """
        seq = sequence[0]

        # embed sequence tokens, get input sequence
        input_seq = self.embedding[0](*seq)  # [N, L, E]

        # shift sequence by one token to right to predict tokens autoregressively
        input_seq = self._prepend_sos_token(input_seq, cls)  # [N, L, E]

        # process input sequence by the Transformer stack, get output sequence
        output_seq = self.transformer(
            src=self._transpose(input_seq),  # [L, N, E], pytorch expects the sequence dimension to be first
            mask=look_ahead_mask(input_seq.shape[1], device=input_seq.device),  # [L, L]
            src_key_padding_mask=self.embedding[0].padding_mask(),  # [N, L]
        )  # [S, N, E]
        output_seq = self._transpose(output_seq)  # [N, L, E]

        # return logits
        return self.head[0](output_seq, *seq)  # [N, L, V]
