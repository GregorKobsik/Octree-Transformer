import torch
import torch.nn as nn
import torch.nn.functional as F

from sinkhorn_transformer import SinkhornTransformer


class SinkhornTransformerModule(nn.Module):
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
        super(SinkhornTransformerModule, self).__init__()

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
        self.transformer_encoder = SinkhornTransformer(
            dim=embed_dim,
            depth=num_layers,
            max_seq_len=num_positions,
            heads=num_heads,
            dim_head=None,
            bucket_size=64,
            causal=True,
        )

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab + 1, bias=False)

    def forward(self, value, depth, pos):
        """
        Expect input as shape:
            value: (N, S)
            depth: (N, S)
            pos: (A, N, S)

        shapes:
            S: sequence length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        batch, seq_len = value.shape  # [N, S]

        # pad input - Sequence length needs to be divisible by bucket size
        pad_len = 64 - (seq_len % 64)
        value = F.pad(input=value, pad=(0, pad_len))
        depth = F.pad(input=depth, pad=(0, pad_len))
        pos = F.pad(input=pos, pad=(0, pad_len))

        # embeddings
        x = self.token_embedding(value)  # [N, S, E]
        x = x + self.depth_embedding(depth)  # [N, S, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[axis])  # [N, S, E]

        # prepend start of sequence token
        sos = torch.ones(batch, 1, self.embed_dim, device=value.device) * self.sos  # [N, 1, E]
        x = torch.cat([sos, x[:, :-1, :]], axis=1)  # [N, S, E]

        # transformer encoder TODO: pass mask, to mask out padding in batched inputs (n > 1)
        x = self.transformer_encoder(x)  # [N, S, E]

        # return logits
        return self.head(x)[:, :seq_len]
