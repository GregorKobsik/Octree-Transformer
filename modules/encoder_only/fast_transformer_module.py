import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask
from fast_transformers.feature_maps import Favor

_attention_map = {
    'fast_full': 'full',
    'fast_linear': 'causal-linear',
    'fast_local': 'local',
    'fast_reformer': 'reformer',
    'fast_favor': 'causal-linear',
    'fast_performer': 'causal-linear',  # legacy
}

_feature_map = {
    'fast_full': None,
    'fast_linear': None,
    'fast_local': None,
    'fast_reformer': None,
    'fast_favor': Favor.factory(),
    'fast_performer': Favor.factory(),  # legacy
}


class FastTransformerModule(nn.Module):
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
        super(FastTransformerModule, self).__init__()
        self.attention_type = _attention_map[attention]

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
        kwargs = {
            'attention_type': _attention_map[attention],
            'local_context': 512,
            'n_layers': num_layers,
            'n_heads': num_heads,
            'feed_forward_dimensions': embed_dim * 4,
            'query_dimensions': embed_dim // num_heads,
            'value_dimensions': embed_dim // num_heads,
            'dropout': 0.0,
            'attention_dropout': 0.0,
            'activation': "gelu",
            'feature_map': _feature_map[attention],
        }
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(**kwargs).get()

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab + 1, bias=False)

    def forward(self, value, depth, pos):
        """
        Expect input as shape:
            value: (N, S)
            depth: (N, S)
            pos: (A, N, S)

        shapes:
            N: batch size
            S: sequence length
            E: embedding dimension
            A: spatial dimension
        """
        batch, seq_len = value.shape  # [N, S]

        if self.attention_type == "reformer":
            pad_len = 128 - (seq_len % 128)
            value = F.pad(input=value, pad=(0, pad_len))
            depth = F.pad(input=depth, pad=(0, pad_len))
            pos = F.pad(input=pos, pad=(0, pad_len))

        # triangular causal and padding masks
        causal_mask = TriangularCausalMask(value.shape[1], device=value.device)  # [S, S]
        padding_mask = FullMask(mask=value != 0, device=value.device)  # [N, S]

        # embeddings
        x = self.token_embedding(value)  # [N, S, E]
        x = x + self.depth_embedding(depth)  # [N, S, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[axis])  # [N, S, E]

        # prepend start of sequence token
        sos = torch.ones(batch, 1, self.embed_dim, device=value.device) * self.sos  # [N, 1, E]
        x = torch.cat([sos, x[:, :-1, :]], axis=1)  # [N, S, E]

        # transformer encoder
        x = self.transformer_encoder(
            x=x,
            attn_mask=causal_mask,
            length_mask=padding_mask,
        )  # [N, S, E]

        # return logits
        return self.head(x)[:, :seq_len]
