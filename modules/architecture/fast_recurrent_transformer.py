import torch
import torch.nn as nn

from fast_transformers.builders import RecurrentEncoderBuilder


class FastRecurrentTransformer(nn.Module):
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
        """ Creates an instance of a fast recurrent transformer module.

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
        super(FastRecurrentTransformer, self).__init__()

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

        # transformer stack
        self.transformer = RecurrentEncoderBuilder.from_kwargs(
            attention_type="full",
            n_layers=num_layers,
            n_heads=num_heads,
            feed_forward_dimensions=4 * embed_dim,
            query_dimensions=embed_dim // num_heads,
            value_dimensions=embed_dim // num_heads,
            dropout=dropout,
            attention_dropout=dropout,
            activation='gelu',
        ).get()

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

        # embed sequence tokens and shift sequence by one token to right
        input_seq = self.token_embedding(seq, cls)  # [N, L, E]

        # process input sequence by the Transformer stack, get output sequence
        memory = torch.zeros_like(input_seq)
        state = None
        for i in range(input_seq.shape[1]):
            memory[:, i], state = self.transformer_module(input_seq[:, i], state=state)

        # return logits
        return self.generative_head(memory, seq)  # [N, L, V]

    def token_embedding(self, x, cls):
        """ Embeds the octree sequence and prepends a class or sos token.

        Args:
            x: Octree sequence ([N, L], [N, L], [N, L, A]).

        Returns:
            Input token sequence.
        """
        # embed sequence tokens, get input sequence
        input_seq = self.embedding[0](*x)  # [N, L, E]

        # shift sequence by one token to right to predict tokens autoregressively
        return self._prepend_sos_token(input_seq, cls)  # [N, L, E]

    def transformer_module(self, x, state):
        """ Performs a single Transformer module operation on the given input

        Args:
            x: A single input sequence token [N, E].
            state: The Transformer state.
        """
        return self.transformer(x, state=state)

    def generative_head(self, x, seq, last_only=True):
        """ Process the generative head once and return logits for given output token sequence.

        Args:
            x: Output token sequence [N, L, E].
            seq: Original octree sequence.
            last_only: Process only last depth layer.

        Returns:
            Logits for each output token.
        """
        return self.head[0](x, *seq, last_only)  # [N, L, V]
