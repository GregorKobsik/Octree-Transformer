import torch
import torch.nn as nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask


class FastShapeTransformerModel(nn.Module):
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
        super(FastShapeTransformerModel, self).__init__()

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
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            attention_type="causal-linear",
            n_layers=num_layers,
            n_heads=num_heads,
            feed_forward_dimensions=embed_dim * 4,
            query_dimensions=embed_dim // num_heads,
            value_dimensions=embed_dim // num_heads,
            dropout=0.0,
            activation="gelu",
        ).get()

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab + 1, bias=False)

    def forward(self, seq, depth, pos):
        """
        Expect input as shape:
            seq: (N, S)
            depth: (N, S)
            pos: (A, N, S)

        shapes:
            N: batch size
            S: sequence length
            E: embedding dimension
            A: spatial dimension
        """
        batch, seq_len = seq.shape  # [N, S]

        # triangular causal and padding masks
        causal_mask = TriangularCausalMask(seq_len, device=seq.device)  # [S, S]
        padding_mask = FullMask(mask=seq != 0, device=seq.device)  # [N, S]

        # embeddings
        x = self.token_embedding(seq)  # [N, S, E]
        x = x + self.depth_embedding(depth)  # [N, S, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[axis])  # [N, S, E]

        # prepend start of sequence token
        sos = torch.ones(batch, 1, self.embed_dim, device=seq.device) * self.sos  # [N, 1, E]
        x = torch.cat([sos, x[:, :-1, :]], axis=1)  # [N, S, E]

        # transformer encoder
        x = self.transformer_encoder(
            x=x,
            attn_mask=causal_mask,
            length_mask=padding_mask,
        )  # [N, S, E]

        # return logits
        return self.head(x)

    def get_attn_weights(self):
        return self.transformer_encoder._attention_weights

    def get_attn_activations(self):
        return self.transformer_encoder._attention_activations
