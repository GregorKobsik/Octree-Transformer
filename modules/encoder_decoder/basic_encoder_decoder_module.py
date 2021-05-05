import torch
import torch.nn as nn


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
        self.num_vocab = num_vocab
        self.spatial_dim = spatial_dim
        self.max_seq_len = num_positions // 2

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # embeddings
        self.token_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2**tree_depth + 1, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # transformer encoder decoder
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers // 2,
            num_decoder_layers=num_layers // 2,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
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

    def _embed(self, value, depth, pos):
        x = self.token_embedding(value)
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[axis])

    def forward(self, src_value, tgt_value, src_depth, tgt_depth, src_pos, tgt_pos):
        """
        Expect input as shape:
            (src/tgt)_value: (S/T, N)
            (src/tgt)_depth: (S/T, N)
            (src/tgt)_pos: (A, S/T, N)

        shapes:
            S: source length
            T: target length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """

        # embeddings -> [S/T, N, E]
        src = self._embed(src_value, src_depth, src_pos)
        tgt = self._embed(tgt_value, tgt_depth, tgt_pos)

        # prepend start of sequence token -> [T, N, E]
        _, batch = tgt_value.shape  # [T, N]
        sos = torch.ones(1, batch, self.embed_dim, device=tgt_value.device) * self.sos  # [1, N, E]
        tgt = torch.cat([sos, tgt[:-1, :, :]], axis=0)

        # transformer encoder
        h = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=self._create_look_ahead_mask(tgt),
            src_key_padding_mask=self._create_padding_mask(src),
            tgt_key_padding_mask=self._create_padding_mask(tgt),
        )  # [T, N, E]

        # return logits
        return self.head(h)
