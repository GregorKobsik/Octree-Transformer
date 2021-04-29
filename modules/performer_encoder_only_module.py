import torch
import torch.nn as nn

from performer_pytorch import Performer


class PerformerEncoderOnlyModule(nn.Module):
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
        super(PerformerEncoderOnlyModule, self).__init__()

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

        # performer encoder
        self.transformer_encoder = Performer(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            causal=True,
            local_attn_heads=num_heads // 2,
            local_window_size=num_positions // 8,
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            reversible=True,
            ff_chunks=10,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            qr_uniform_q=False,
            use_scalenorm=False,
            use_rezero=False,
            ff_glu=True,
            ff_dropout=0.,
            attn_dropout=0.,
            cross_attend=False,
            no_projection=False,
            auto_check_redraw=True,
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

        # transformer encoder TODO: pass mask, to mask out padding in batched inputs (n > 1)
        h = self.transformer_encoder(h)  # [S, N, E]

        # return logits
        return self.head(h)
