import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.src_token_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.tgt_token_embedding = nn.Embedding(num_vocab**2**self.spatial_dim + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2**tree_depth + 1, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )  # TODO: + 1 unneccessary

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

        # transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation='gelu',
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # final linear layer
        self.head = nn.Linear(embed_dim, num_vocab**2**self.spatial_dim + 1, bias=False)

    def _embed(self, x, depth, pos):
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(pos[:, :, axis])
        return x

    def encode(self, value, depth, pos):
        """
        Expect input as shape:
            value: (S, N)
            depth: (S, N)
            pos: (S, N, A)

        shapes:
            S: source length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        # embeddings -> [S, N, E]
        x = self.src_token_embedding(value)  # [S, N, E]
        src = self._embed(x, depth, pos)  # [S, N, E]

        # encoder part of the transformer - compute memory
        return self.transformer_encoder(src, src_key_padding_mask=padding_mask(value, device=src.device))

    def decode(self, target, depth, pos, memory):
        """
        Expect input as shape:
            target: (T, N)
            depth: (S, N)
            pos: (S, N, A)

        shapes:
            T: target length
            N: batch size
            E: embedding dimension
            A: spatial dimension
        """
        tgt_len, batch = target.shape  # [T, N]

        # embeddings -> [T, N, E]
        x = self.tgt_token_embedding(target)  # [T, N, E]
        tgt = self._embed(x, depth, pos)  # [T, N, E]

        # prepend start of sequence token -> [T, N, E]
        sos = torch.ones(1, batch, self.embed_dim, device=tgt.device) * self.sos  # [1, N, E]
        tgt = torch.cat([sos, tgt[:-1]], axis=0)  # [T, N, E]

        # decoder part of the transformer
        h = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=look_ahead_mask(tgt_len, device=tgt.device),
            tgt_key_padding_mask=padding_mask(target, device=tgt.device),
        )  # [T, N, E]

        # return logits: [T, N, E] -> [T, num_vocab**2**spatial_dim]
        return self.head(h)  # [T, num_vocab**2**spatial_dim]

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
        # extract valid target sequence, if target is longer than last layer in input
        tgt_len, _ = target.shape  # [T, N]
        tgt_idx = torch.argmax(depth)
        max_tgt_len = len(depth[tgt_idx:])
        tgt_depth = depth[tgt_idx:tgt_idx + tgt_len] + 1  # [T, N]
        tgt_pos = pos[tgt_idx:tgt_idx + tgt_len]  # [T, N, A]

        # transformer encoder decoder
        memory = self.encode(value, depth, pos)  # [T, N, E]
        output = self.decode(target[:max_tgt_len], tgt_depth, tgt_pos, memory)  # [T, num_vocab**2**spatial_dim]

        # pad output if target sequence was extracted
        logits = F.pad(output, pad=(0, 0, 0, 0, 0, tgt_len - max_tgt_len), mode='constant', value=0)

        # return logits
        return logits
