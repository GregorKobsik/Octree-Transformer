from argparse import ArgumentParser

import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from models import ShapeTransformerModel
from lr_scheduler import CosineAnnealingWarmupRestarts


class ShapeTransformer(pl.LightningModule):
    def __init__(
        self,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_positions=512,
        num_vocab=16,
        learning_rate=3e-3,
        warmup_steps=500,
        train_steps=10_000,
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
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps

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
        """ Adam optimizer with cosine annealing and warmup learning rate scheduler. """
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler":
                CosineAnnealingWarmupRestarts(
                    optimizer,
                    self.train_steps,
                    max_lr=self.learning_rate,
                    min_lr=0.0,
                    warmup_steps=self.warmup_steps,
                ),
            "interval":
                "step",
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
