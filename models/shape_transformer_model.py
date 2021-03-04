import torch
import torch.nn as nn

from layers import TransformerEncoderLayer, TransformerEncoder


class ShapeTransformerModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        num_positions,
        num_vocab,
        spatial_dim,
        tree_depth,
    ):
        super(ShapeTransformerModel, self).__init__()

        self.embed_dim = embed_dim
        self.num_vocab = num_vocab
        self.spatial_dim = spatial_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # embeddings
        self.token_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2**tree_depth + 1, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab + 1, bias=False)

    def _create_look_ahead_mask(self, x):
        """ Creates a diagonal mask, which prevents the self-attention to look ahead. """
        seq_len, _ = x.shape  # [S, N]
        attn_mask = torch.full((seq_len, seq_len), -float("Inf"), device=x.device)  # [S, S]
        return torch.triu(attn_mask, diagonal=1)  # [S, S]

    def _create_padding_mask(self, x):
        """ Create a padding mask for the given input.

        Always assumens '0' as a padding value.

        PyTorch Transformer defines 'src_key_padding_mask' with shape (N, S),
        where the input shape of 'src' is (S, N, E).
        Therefor we need to transpose the dimensions of the created mask.
        """
        mask = torch.zeros_like(x, device=x.device).masked_fill(x == 0, 1).bool()  # [S, N]
        return mask.transpose(0, 1)  # [N, S]

    def forward(self, seq, depth, pos):
        """
        Expect input as shape:
            seq: (S, N)
            depth: (S, N)
            pos: (A, S, N)

        shapes:
            S: sequence length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        # look-ahead and padding masks
        look_ahead_mask = self._create_look_ahead_mask(seq)  # [S, S]
        padding_mask = self._create_padding_mask(seq)  # [N, S]

        # embeddings
        h = self.token_embedding(seq)  # [S, N, E]
        h = h + self.depth_embedding(depth)  # [S, N, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            h = h + spatial_embedding(pos[axis])  # [S, N, E]

        # prepend start of sequence token
        _, batch = seq.shape  # [S, N]
        sos = torch.ones(1, batch, self.embed_dim, device=seq.device) * self.sos  # [1, N, E]
        h = torch.cat([sos, h[:-1, :, :]], axis=0)  # [S, N, E]

        # transformer encoder
        h = self.transformer_encoder(
            src=h,
            mask=look_ahead_mask,
            src_key_padding_mask=padding_mask,
        )  # [S, N, E]

        # return logits
        return self.head(h)

    def get_attn_weights(self):
        return self.transformer_encoder._attention_weights

    def get_attn_activations(self):
        return self.transformer_encoder._attention_activations
