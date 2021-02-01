import torch
import torch.nn as nn


class ShapeTransformerModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        num_positions,
        num_vocab,
    ):
        super(ShapeTransformerModel, self).__init__()

        print("Shape Transformer model parameters:")
        print("- Embedding dimension:", embed_dim)
        print("- Number of attention heads:", num_heads)
        print("- Number of decoder layers:", num_layers)
        print("- Maximal sequence length:", num_positions)
        print("- Size of vocabulary:", num_vocab)
        print()

        self.embed_dim = embed_dim
        self.num_vocab = num_vocab

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # embeddings
        self.token_embeddings = nn.Embedding(num_vocab, embed_dim, padding_idx=self.num_vocab - 1)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

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

        self.head = nn.Linear(embed_dim, num_vocab, bias=False)

    def _create_look_ahead_mask(self, x):
        """ Creates a diagonal mask, which prevents the self-attention to look ahead. """
        seq_len, _ = x.shape  # [S, N]
        attn_mask = torch.full((seq_len, seq_len), -float("Inf"), device=x.device)  # [S, S]
        return torch.triu(attn_mask, diagonal=1)  # [S, S]

    def _create_padding_mask(self, x):
        """ Create a padding mask for the given input.

        PyTorch Transformer defines 'src_key_padding_mask' with shape (N, S),
        where the input shape of 'src' is (S, N, E).
        Therefor we need to transpose the dimensions of the created mask.
        """
        # TODO: select a more generic padding value, e.g. `<pad>` if possible
        mask = torch.zeros_like(x, device=x.device).masked_fill(x == self.num_vocab - 1, 1).bool()  # [S, N]
        return mask.transpose(0, 1)  # [N, S]

    def forward(self, x):
        """
        Expect input as shape (S, N)

        shapes:
            S: sequence length
            N: batch size
            E: embedding dimension
        """
        # look-ahead and padding masks
        look_ahead_mask = self._create_look_ahead_mask(x)  # [S, S]
        padding_mask = self._create_padding_mask(x)  # [N, S]

        # embeddings
        length, batch = x.shape  # [S, N]
        h = self.token_embeddings(x)  # [S, N, E]
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos  # [1, N, E]
        h = torch.cat([sos, h[:-1, :, :]], axis=0)  # [S, N, E]
        positions = torch.arange(length, device=x.device).unsqueeze(-1)  # [S, 1]
        h = h + self.position_embeddings(positions).expand_as(h)  # [S, N, E]

        # transformer encoder
        h = self.transformer_encoder(
            src=h,
            mask=look_ahead_mask,
            src_key_padding_mask=padding_mask,
        )  # [S, N, E]

        logits = self.head(h)  # [S, N, E]
        return logits
