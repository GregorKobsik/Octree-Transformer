import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models import ShapeTransformerModel


class ShapeTransformer(pl.LightningModule):
    def __init__(
        self,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_positions=512,
        num_vocab=16,
        learning_rate=3e-3,
        steps=10_000,
        warmup_steps=500,
        **kwargs,
    ):
        super(ShapeTransformer, self).__init__()
        self.save_hyperparameters()
        self.model = ShapeTransformerModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_positions,
            num_vocab=num_vocab,
        )

        self.num_positions = num_positions
        self.num_vocab = num_vocab
        self.loss_criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.steps = steps
        self.warmup_steps = warmup_steps

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument("--num_positions", type=int, default=512)
        parser.add_argument("--num_vocab", type=int, default=16)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=3e-3)
        parser.add_argument("--epochs", type=int, default=50)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        scheduler = {
            "scheduler": LambdaLR(optimizer, learning_rate_schedule(self.warmup_steps, self.steps)),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        x, y = batch
        x = x[:self.num_positions, :].long()

        logits = self.model(x)
        loss = self.loss_criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss


def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""
    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn
