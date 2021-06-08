from argparse import ArgumentParser

import torch
from torch.optim import Adam
import pytorch_lightning as pl

from modules.transformer import (
    BasicTransformer,
)
from modules.embedding import (
    BasicEmbedding,
    SingleConvolutionalEmbeddingA,
    SingleConvolutionalEmbeddingB,
    SingleConvolutionalEmbeddingC,
    SingleConvolutionalEmbeddingD,
    DoubleConvolutionalEmbedding,
)
from modules.generative_head import (
    LinearHead,
    SingleConvolutionalHeadA,
    SingleConvolutionalHeadB,
    DoubleConvolutionalHead,
)
from lr_scheduler import (
    ConstantWithWarmup,
)
from loss import (
    CrossEntropyLoss,
)


class ShapeTransformer(pl.LightningModule):
    """ Creates a shape transformer modell.

    Defines an abstract shell for different implementations of the shape transformer to allow for an experimental
    development of different modules. A loose and modular definition allows to exchange key components without
    reimplementing the whole training process, while defining a clean API for the data input and training process.

    Args:
        embed_dim: Number of embedding dimensions used by `attention`.
        num_heads: Number of heads used by `attention`.
        num_layers: Number of layers for each the 'decoder' and 'encoder' part of the transformer.
        num_positions: Maximal length of processed input tokens for the 'decoder' and 'encoder'. You can pass longer
            sequences as input, but they will be truncated before feeding into the transformer. Although longer
            sequences can be accepted by a non-basic embedding and possibly compressed to stay within the limit.
        num_vocab: Number of different used vocabs in the vocabulary set.
        resolution: Maximum side length of input data.
        spatial_dim: Spatial dimensionality of input data.
        learning_rate: Maximum learning rate used durring the training process.
        warmup_steps: Number of steps used for a warmup, where the learning rate is increasing linearily. It can be
            defined as an integer to define an absolute number of training steps or as a floating point number to
            define a relative number or used warmup steps in the range [0 .. 1].
        train_steps: Maximum number of training steps used during training.
        loss_function: Defines the loss function used for training.
        architecture: Defines the base architecture of the transformer, either 'encoder_only' or 'encoder_decoder'
        attention: Defines the used attention implementation in the transformer.
        embedding: Defines the used token embedding of the shape transformer.
        head: Defines the used generative head of the shape transformer.
    """
    def __init__(
        self,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_positions=512,
        num_vocab=16,
        resolution=32,
        spatial_dim=2,
        learning_rate=3e-3,
        warmup_steps=500,
        train_steps=10_000,
        loss_function='cross_entropy',
        architecture='encoder_only',
        attention='basic_full',
        embedding='basic',
        head='generative_basic',
        **kwargs,
    ):
        super(ShapeTransformer, self).__init__()
        self.save_hyperparameters()

        # token embedding
        if embedding == 'basic':
            embedding = BasicEmbedding(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding in ('single_conv', 'single_conv_A'):
            embedding = SingleConvolutionalEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_B':
            embedding = SingleConvolutionalEmbeddingB(embed_dim, spatial_dim)
        elif embedding == 'single_conv_C':
            embedding = SingleConvolutionalEmbeddingC(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_D':
            embedding = SingleConvolutionalEmbeddingD(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'double_conv':
            embedding = DoubleConvolutionalEmbedding(embed_dim, spatial_dim)
        else:
            print(f"ERROR: {embedding} embedding not implemented.")
            raise ValueError

        # generative head
        if head == "generative_basic":
            head = LinearHead(num_vocab, embed_dim)
        elif head in ('single_conv', 'single_conv_A'):
            head = SingleConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
        elif head == 'single_conv_B':
            head = SingleConvolutionalHeadB(num_vocab, embed_dim, spatial_dim)
        elif head == 'double_conv':
            head = DoubleConvolutionalHead(num_vocab, embed_dim, spatial_dim)
        else:
            print(f"ERROR: {head} head not implemented.")
            raise ValueError

        # transformer model
        kwargs = {
            'token_embedding': embedding,
            'generative_head': head,
            'architecture': architecture,
            'attention': attention,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'num_positions': num_positions,
            'num_vocab': num_vocab,
            'resolution': resolution,
            'spatial_dim': spatial_dim,
        }
        if attention == "basic":
            self.model = BasicTransformer(**kwargs)
        else:
            print(f"ERROR: {attention} attention not implemented.")
            raise ValueError

        # loss function
        if loss_function == 'cross_entropy':
            self.loss_function = CrossEntropyLoss()
        else:
            print(f"ERROR: {loss_function} loss not implemented.")
            raise ValueError

        print(f"\nShape Transformer parameters:\n{self.hparams}\n")

    # TODO: check if needed
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

    def forward(self, sequence):
        """ Performs a full transformer pass of the input sequence.

        Args:
            sequence: Tuple containing different input sequences for the 'encoder_only' and 'encoder_decoder'
                architecture. The 'encoder_only' architecture expects sequence to be a tuple of (value, depth, position)
                sequences, while the 'encoder_decoder' architecture expects sequence to the a tuple of
                (encoder_sequence, decoder_sequence) inputs for the encoder and decoder, respectively.

        Return:
            Logits which describe the autoregressive likelihood of the next target token.
        """
        return self.model(sequence)

    def step(self, batch):
        """ Perform one full transformer pass and compute the loss.

        Args:
            batch: Holds batched input sequences.

        Return:
            Return the loss value for the given batch.
        """
        sequence, target = batch
        logits = self.forward(sequence)
        target = target[:, :logits.shape[1]].contiguous()  # limit target tokens, if we had to limit input size
        loss = self.loss_function(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        """ Perform one training step with the given batch and log the loss. """
        loss = self.step(batch)
        self.log('loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Perform one validation step with the given batch and log the loss as well the allocated memory. """
        loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('mem_alloc', torch.cuda.max_memory_allocated() / 1024**2, prog_bar=True, sync_dist=True)
        return loss
