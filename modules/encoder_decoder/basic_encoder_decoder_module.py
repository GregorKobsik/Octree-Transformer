import torch
import torch.nn as nn

from masks import look_ahead_mask, padding_mask


class BasicEncoderDecoderModule(nn.Module):
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
        super(BasicEncoderDecoderModule, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vocab = num_vocab
        self.spatial_dim = spatial_dim
        self.attention = attention

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # embeddings
        self.token_embedding = nn.Embedding(num_vocab**2**self.spatial_dim + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2**tree_depth + 1, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # transformer encoder decoder
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab**2**self.spatial_dim + 1, bias=False)

    def _embed(self, value, depth, pos):
        x = self.token_embedding(value)
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[:, :, axis])
        return x

    def forward(self, value, depth, pos, target):
        """
        Expect input as shape:
            value: (S, N)
            depth: (S, N)
            pos: (S, N, A)
            target: (T, N)

        shapes:
            S: source length
            T: target length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        src_len, batch = value.shape  # [S, N]
        tgt_len, _ = target.shape  # [T, N]

        # embeddings -> [S/T, N, E]
        src = self._embed(value, depth, pos)  # [S, N, E]
        tgt = self._embed(target, depth[-tgt_len:] + 1, pos[-tgt_len:])  # [T, N, E]

        # prepend start of sequence token -> [T, N, E]
        sos = torch.ones(1, batch, self.embed_dim, device=tgt.device) * self.sos  # [1, N, E]
        tgt = torch.cat([sos, tgt[:-1, :, :]], axis=0)  # [T, N, E]

        # encoder decoder transformer
        h = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=look_ahead_mask(tgt_len, device=tgt.device),
            src_key_padding_mask=padding_mask(value, device=src.device),
            tgt_key_padding_mask=padding_mask(target, device=tgt.device),
        )  # [T, N, E]

        # return logits: [T, N, E] -> [T, num_vocab**2**spatial_dim]
        return self.head(h)  # [T, num_vocab**2**spatial_dim]
