from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from models.basic_shape_transformer_model import BasicShapeTransformerModel
from models.fast_shape_transformer_model import FastShapeTransformerModel
from lr_scheduler import ConstantWithWarmup


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

        if attention.startswith('basic'):
            self.model = BasicShapeTransformerModel(**kwargs)
            self.batch_first = False
        elif attention.startswith('fast'):
            self.model = FastShapeTransformerModel(**kwargs)
            self.batch_first = True
        else:
            print(f"ERROR: No configuration available for attention: {attention}.")
            raise ValueError
        print(f"\nShape Transformer parameters:\n{self.hparams}\n")

        if loss_function == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            print(f"ERROR: No configuration available for loss_function: {loss_function}.")
            raise ValueError

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

    def forward(self, value, depth, pos):
        return self.model(value, depth, pos)

    def step(self, batch, batch_idx):
        value, depth, pos = batch

        # input lenght delimited to 'num_positions'
        value = value[:self.hparams.num_positions].long()
        depth = depth[:self.hparams.num_positions].long()
        pos = pos[:, :self.hparams.num_positions].long()

        # 'fast-transformers' expects, unlike 'torch', batch size first and the sequence second.
        if self.batch_first:
            value = torch.transpose(value, 0, 1).contiguous()
            depth = torch.transpose(depth, 0, 1).contiguous()
            pos = torch.transpose(pos, 1, 2).contiguous()

        logits = self.model(value, depth, pos)
        loss = self.loss_function(logits.view(-1, logits.size(-1)), value.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True)
        return loss
