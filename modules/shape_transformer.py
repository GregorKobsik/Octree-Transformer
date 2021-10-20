import math
import torch
from torch.optim import Adam
import pytorch_lightning as pl

from .token_embedding import create_embedding
from .generative_head import create_head
from .architecture import create_architecture

from utils import nanmean
from utils.loss import create_loss
from utils.lr_scheduler import ConstantWithWarmup


class ShapeTransformer(pl.LightningModule):
    """ Creates a shape transformer modell.

    Defines an abstract shell for different implementations of the shape transformer to allow for an experimental
    development of different modules. A loose and modular definition allows to exchange key components without
    reimplementing the whole training process, while defining a clean API for the data input and training process.

    Args:
        embed_dim: Size of embedding dimensions used by `attention`.
        num_heads: Number of heads used by `attention`.
        num_layers: Number of layers for each the 'decoder' and 'encoder' part of the transformer.
        num_positions: Maximal length of processed input tokens for the 'decoder' and 'encoder'. You can pass longer
            sequences as input, but they will be truncated before feeding into the transformer. Although longer
            sequences can be accepted by a non-basic embedding and possibly compressed to stay within the limit.
        num_vocab: Number of different vocabs in the vocabulary set.
        resolution: Maximum side length of input data.
        spatial_dim: Spatial dimensionality of input data.
        learning_rate: Maximum learning rate used durring the training process.
        dropout: The dropout rate.
        warmup_steps: Number of steps used for a warmup, where the learning rate is increasing linearily. It can be
            defined as an integer to define an absolute number of training steps or as a floating point number to
            define a relative number or used warmup steps in the range [0 .. 1].
        train_steps: Maximum number of training steps used during training.
        loss_function: Defines the loss function used for training.
        val_loss_function: Defines the loss function used for validation.
        architecture: Defines the base architecture of the transformer, either 'encoder_only' or 'encoder_decoder'
        attention: Defines the used attention implementation in the transformer.
        token_encoding: Defines the used token embedding of the shape transformer.
        embedding: Defines the used token reduction of the shape transformer.
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
        dropout=0.1,
        warmup_steps=500,
        train_steps=10_000,
        loss_function='cross_entropy',
        val_loss_function='cross_entropy',
        architecture='encoder_only',
        attention='basic_full',
        token_encoding='basic',
        embedding='basic',
        head='generative_basic',
        transform='linear_max_res',
        **kwargs,
    ):
        super(ShapeTransformer, self).__init__()
        self.save_hyperparameters()
        self.resolution = resolution

        # token embedding
        embedding = create_embedding(
            name=embedding,
            token_encoding=token_encoding,
            num_vocab=num_vocab,
            embed_dim=embed_dim,
            resolution=resolution,
            spatial_dim=spatial_dim,
        )

        # generative head
        head = create_head(
            name=head,
            num_vocab=num_vocab,
            embed_dim=embed_dim,
            resolution=resolution,
            spatial_dim=spatial_dim,
        )

        # transformer model
        self.model = create_architecture(
            architecture=architecture,
            attention=attention,
            token_embedding=embedding,
            generative_head=head,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_positions,
            dropout=dropout,
        )

        # training loss function
        self.loss_function = create_loss(
            name=loss_function,
            ignore_index=0,
            max_depth=math.log2(resolution),
            spatial_dim=spatial_dim,
        )

        # validation loss function
        self.val_loss_function = create_loss(
            name=val_loss_function,
            ignore_index=0,
            max_depth=math.log2(resolution),
            spatial_dim=spatial_dim,
        )

        print(f"\nShape Transformer parameters:\n{self.hparams}\n")

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

    def training_step(self, batch, batch_idx):
        """ Perform one training step with the given batch and log the loss. """
        sequence, target = batch

        logits = self.forward(sequence)
        loss = self.compute_and_log_loss(logits, target, self.loss_function, prefix='training/train_')
        self.compute_and_log_loss(logits, target, self.val_loss_function, prefix='training/val_')

        return loss

    def validation_step(self, batch, batch_idx):
        """ Perform one validation step with the given batch and log the loss as well the allocated memory. """
        sequence, target = batch

        logits = self.forward(sequence)
        self.compute_and_log_loss(logits, target, self.loss_function, prefix='validation/train_', log_per_layer=True)
        loss = self.compute_and_log_loss(
            logits, target, self.val_loss_function, prefix='validation/val_', log_per_layer=True
        )

        # log allocated memory
        self.log('mem_alloc', torch.cuda.max_memory_allocated() / 1024**2, sync_dist=True)
        self.log('mem_reserv', torch.cuda.max_memory_reserved() / 1024**2, sync_dist=True)

        return loss

    def compute_and_log_loss(self, logits, target, loss_fx, prefix="", log_per_layer=False):
        """ Compute mean loss and per layer loss and log it. Return mean loss. """
        tgt_val, tgt_dep, tgt_pos = target

        # limit target tokens, if we had to limit input size
        tgt_val = tgt_val[:, :logits.shape[1]]
        tgt_dep = tgt_dep[:, :logits.shape[1]]
        tgt_pos = tgt_pos[:, :logits.shape[1]]

        # compute loss for each token
        loss = loss_fx(logits, (tgt_val, tgt_dep, tgt_pos))

        # compute mean loss
        mean_loss = torch.mean(loss[tgt_dep != 0])
        self.log(prefix + 'loss_mean', mean_loss, sync_dist=True)

        # compute loss per layer
        loss_per_layer = []
        for i in range(2, int(math.log2(self.resolution) + 1)):
            layer_loss = torch.mean(loss[tgt_dep == i])
            loss_per_layer += [layer_loss]
            if log_per_layer:
                self.log('per_layer_loss/' + prefix + f'loss_layer_{i}', layer_loss, sync_dist=True, reduce_fx=nanmean)
        mean_loss_per_layer = nanmean(torch.tensor(loss_per_layer, device=loss.device))
        self.log(prefix + 'loss_layer_mean', mean_loss_per_layer, sync_dist=True, reduce_fx=nanmean)

        return mean_loss
