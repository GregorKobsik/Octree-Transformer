import torch
import torch.nn as nn

from masks import look_ahead_mask, padding_mask, ancestor_mask


class BasicEncoderOnlyModule(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        num_positions,
        num_vocab,
        spatial_dim,
        tree_depth,
        attention,
    ):
        super(BasicEncoderOnlyModule, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vocab = num_vocab
        self.spatial_dim = spatial_dim
        self.attention = attention

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

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab + 1, bias=False)

    def forward(self, value, depth, pos):
        """
        Expect input as shape:
            value: (S, N)
            depth: (S, N)
            pos: (A, S, N)

        shapes:
            S: sequence length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        # embeddings
        h = self.token_embedding(value)  # [S, N, E]
        h = h + self.depth_embedding(depth)  # [S, N, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            h = h + spatial_embedding(pos[axis])  # [S, N, E]

        # prepend start of sequence token
        seq_len, batch = value.shape  # [S, N]
        sos = torch.ones(1, batch, self.embed_dim, device=value.device) * self.sos  # [1, N, E]
        h = torch.cat([sos, h[:-1, :, :]], axis=0)  # [S, N, E]

        # create attention mask
        if self.attention == "basic_ancestor":
            attn_mask = ancestor_mask(value, h.device)
            attn_mask = torch.repeat_interleave(attn_mask, self.num_heads, dim=0)
        else:
            attn_mask = look_ahead_mask(seq_len, device=h.device)

        # transformer encoder
        h = self.transformer_encoder(
            src=h,
            mask=attn_mask,
            src_key_padding_mask=padding_mask(value, device=h.device),
        )  # [S, N, E]

        # return logits
        return self.head(h)

    def get_attn_weights(self):
        return self.transformer_encoder._attention_weights

    def get_attn_activations(self):
        return self.transformer_encoder._attention_activations
