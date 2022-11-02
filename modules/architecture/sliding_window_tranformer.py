import torch
import torch.nn as nn

from .pytorch_transformer import PytorchTransformer
from local_attention import LocalAttention


class SlidingWindowTransformer(PytorchTransformer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        window_size,
        token_embedding,
        generative_head,
        dropout,
        num_classes,
        **_,
    ):
        """ Creates an instance of a transformer module using the local attention head.
        Local Attention: https://github.com/lucidrains/local-attention

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
            window_size: Size of the window, which each head is allowed to look at.
            token_embedding: Instance of an embedding layer, which embedds given sequences of tokens into an embedding
                space, which is the direct input for the transformer layers.
            generative_head: Instance of a head layer, which transforms the output of the transformer into logits.
            dropout: The dropout value.
            num_classes: If bigger than one, the transformer will be class conditional
        """
        super(SlidingWindowTransformer, self).__init__(
            embed_dim,
            num_heads,
            num_layers,
            token_embedding,
            generative_head,
            dropout,
            num_classes,
        )

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
        encoder_layer = SlidingWindowEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            window_size=window_size,
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


class SlidingWindowEncoderLayer(nn.TransformerEncoderLayer):
    """
    https://github.com/lucidrains/local-attention
    """
    def __init__(self, d_model, nhead, dim_feedforward, window_size, dropout, activation):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.window_size = window_size
        self.self_attn = LocalAttention(
            dim=d_model,
            window_size=window_size,
            causal=True,
            look_backward=1,
            look_forward=0,
            dropout=dropout,
            exact_windowsize=True,
        )

    def forward(self, src, src_mask, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (unused).
            src_key_padding_mask: the mask for the src keys per batch (unused).

        Shape:
            see the docs in Transformer class.
        """
        src = src.transpose(0, 1)
        b, t, e = src.shape
        reminder = (self.window_size - t % self.window_size) % self.window_size
        src = torch.cat([src, torch.zeros(b, reminder, e, device=src.device)], dim=1)

        src2 = self.self_attn(src, src, src, input_mask=None)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.transpose(0, 1)[:t]
