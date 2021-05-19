from argparse import ArgumentParser

import torch
from torch.optim import Adam
import pytorch_lightning as pl

from modules.encoder_only import (
    BasicTransformerModule,
    FastTransformerModule,
    PerformerModule,
    ReformerModule,
    RoutingTransformerModule,
    SinkhornTransformerModule,
    LinearTransformerModule,
)
from modules.encoder_decoder import (
    BasicEncoderDecoderModule,
)
from lr_scheduler import ConstantWithWarmup
from loss import CrossEntropyLoss, DescendantWeightedCrossEntropyLoss


class ShapeTransformer(pl.LightningModule):
    def __init__(
        self,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_positions=512,
        num_vocab=16,
        spatial_dim=2,
        tree_depth=6,
        learning_rate=3e-3,
        warmup_steps=500,
        train_steps=10_000,
        attention='basic_full',
        loss_function='cross_entropy',
        architecture='encoder_only',
        **kwargs,
    ):
        super(ShapeTransformer, self).__init__()
        self.save_hyperparameters()
        kwargs = {
            'attention': attention,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'num_positions': num_positions,
            'num_vocab': num_vocab,
            'spatial_dim': spatial_dim,
            'tree_depth': tree_depth,
        }

        # encoder only architectures
        if architecture == 'encoder_only':
            self.max_seq_len = num_positions
            self.is_encoder_decoder = False
            self.batch_first = True
            if attention.startswith('basic'):
                self.model = BasicTransformerModule(**kwargs)
                self.batch_first = False
            elif attention.startswith('fast'):
                self.model = FastTransformerModule(**kwargs)
            elif attention.startswith('performer'):
                self.model = PerformerModule(**kwargs)
            elif attention.startswith('reformer'):
                self.model = ReformerModule(**kwargs)
            elif attention.startswith('routing'):
                self.model = RoutingTransformerModule(**kwargs)
            elif attention.startswith('sinkhorn'):
                self.model = SinkhornTransformerModule(**kwargs)
            elif attention.startswith('linear'):
                self.model = LinearTransformerModule(**kwargs)
            else:
                print(f"ERROR: {attention} for {architecture} not implemented.")
                raise ValueError
        # encoder decoder architectures
        elif architecture == "encoder_decoder":
            self.max_seq_len = num_positions
            self.is_encoder_decoder = True
            self.batch_first = True
            if attention.startswith('basic'):
                self.model = BasicEncoderDecoderModule(**kwargs)
                self.batch_first = False
            else:
                print(f"ERROR: {attention} attention for {architecture} not implemented.")
                raise ValueError
        else:
            print(f"ERROR: {architecture} architecture not implemented.")
            raise ValueError

        # loss functions
        if loss_function == 'cross_entropy':
            self.loss_function = CrossEntropyLoss()
        elif loss_function == 'descendant_weighted_cross_entropy':
            self.loss_function = DescendantWeightedCrossEntropyLoss(spatial_dim=spatial_dim)
        else:
            print(f"ERROR: {loss_function} loss not implemented.")
            raise ValueError

        print(f"\nShape Transformer parameters:\n{self.hparams}\n")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument("--num_positions", type=int, default=512)
        parser.add_argument("--num_vocab", type=int, default=16)
        parser.add_argument("--spatial_dim", type=int, default=2)
        parser.add_argument("--tree_depth", type=int, default=6)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=3e-3)
        parser.add_argument("--epochs", type=int, default=50)
        return parser

    def configure_optimizers(self):
        """ Adam optimizer with cosine annealing and warmup learning rate scheduler. """
        optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler":
                ConstantWithWarmup(
                    optimizer,
                    self.hparams.train_steps,
                    max_lr=self.hparams.learning_rate,
                    min_lr=0.0,
                    warmup_steps=self.hparams.warmup_steps,
                ),
            "interval":
                "step",
        }
        return [optimizer], [scheduler]

    def forward(self, value, depth, pos, target):
        if self.is_encoder_decoder:
            return self.model(value, depth, pos, target)
        else:
            return self.model(value, depth, pos)

    def step(self, batch, batch_idx):
        with torch.no_grad():
            # input lenght delimited to 'num_positions'
            batch = self._limit_sequence_length(batch)
            # 'pytorch' expects, unlike all other libraries, the batch in the second dimension.
            if not self.batch_first:
                batch = self._transpose_sequence(batch)
            value, depth, pos, target = batch

        logits = self.forward(value, depth, pos, target)
        loss = self.loss_function(logits.view(-1, logits.size(-1)), target.view(-1), depth.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('mem_alloc', torch.cuda.max_memory_allocated() / 1024**2, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def _transpose_sequence(self, batch):
        return [torch.transpose(x, 0, 1).contiguous() for x in batch]

    def _limit_sequence_length(self, batch):
        return [x[:, :self.max_seq_len].long() for x in batch]
