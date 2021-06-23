import math
import torch
from torch.optim import Adam
import pytorch_lightning as pl
from utils import nanmean

from modules.transformer import (
    BasicTransformer,
)
from modules.embedding import (
    BasicEmbedding,
    SingleConvolutionalEmbeddingA,
    SingleConvolutionalEmbeddingB,
    SingleConvolutionalEmbeddingC,
    SingleConvolutionalEmbeddingD,
    SingleConvolutionalEmbeddingE,
    SingleConvolutionalEmbeddingF,
    SingleConvolutionalEmbeddingG,
    SingleConvolutionalEmbeddingH,
    SingleConvolutionalEmbeddingI,
    ConcatEmbeddingA,
    ConcatEmbeddingB,
    ConcatEmbeddingC,
    DoubleConvolutionalEmbedding,
)
from modules.generative_head import (
    LinearHead,
    SingleConvolutionalHeadA,
    SingleConvolutionalHeadB,
    SingleConvolutionalHeadC,
    SingleConvolutionalHeadD,
    SplitHeadA,
    SplitHeadB,
    DoubleConvolutionalHead,
)
from lr_scheduler import (
    ConstantWithWarmup,
)
from loss import (
    CrossEntropyLoss,
    DepthWeightedCrossEntropyLossA,
    DepthWeightedCrossEntropyLossB,
    DepthWeightedCrossEntropyLossC,
    DepthWeightedCrossEntropyLossD,
    DepthWeightedCrossEntropyLossE,
    DepthWeightedCrossEntropyLossF,
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
        val_loss_function: Defines the loss function used for validation.
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
        val_loss_function='cross_entropy',
        architecture='encoder_only',
        attention='basic_full',
        embedding='basic',
        head='generative_basic',
        **kwargs,
    ):
        super(ShapeTransformer, self).__init__()
        self.save_hyperparameters()
        self.resolution = resolution

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
        elif embedding == 'single_conv_E':
            embedding = SingleConvolutionalEmbeddingE(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_F':
            embedding = SingleConvolutionalEmbeddingF(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_G':
            embedding = SingleConvolutionalEmbeddingG(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_H':
            embedding = SingleConvolutionalEmbeddingH(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'single_conv_I':
            embedding = SingleConvolutionalEmbeddingI(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'concat_A':
            embedding = ConcatEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'concat_B':
            embedding = ConcatEmbeddingB(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'concat_C':
            embedding = ConcatEmbeddingC(num_vocab, embed_dim, resolution, spatial_dim)
        elif embedding == 'double_conv':
            embedding = DoubleConvolutionalEmbedding(embed_dim, spatial_dim)
        else:
            print(f"ERROR: {embedding} embedding not implemented.")
            raise ValueError

        # generative head
        if head in ('generative_basic', 'linear'):
            head = LinearHead(num_vocab, embed_dim)
        elif head in ('single_conv', 'single_conv_A'):
            head = SingleConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
        elif head == 'single_conv_B':
            head = SingleConvolutionalHeadB(num_vocab, embed_dim, spatial_dim)
        elif head == 'single_conv_C':
            head = SingleConvolutionalHeadC(num_vocab, embed_dim, spatial_dim)
        elif head == 'single_conv_D':
            head = SingleConvolutionalHeadD(num_vocab, embed_dim, spatial_dim)
        elif head == 'split_A':
            head = SplitHeadA(num_vocab, embed_dim, spatial_dim)
        elif head == 'split_B':
            head = SplitHeadB(num_vocab, embed_dim, spatial_dim)
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
        kwargs = {
            'ignore_index': 0,
            'max_depth': math.log2(resolution),
            'spatial_dim': spatial_dim,
        }

        if loss_function == 'cross_entropy':
            self.loss_function = CrossEntropyLoss(**kwargs)
        elif loss_function == 'depth_cross_entropy_A':
            self.loss_function = DepthWeightedCrossEntropyLossA(**kwargs)
        elif loss_function == 'depth_cross_entropy_B':
            self.loss_function = DepthWeightedCrossEntropyLossB(**kwargs)
        elif loss_function == 'depth_cross_entropy_C':
            self.loss_function = DepthWeightedCrossEntropyLossC(**kwargs)
        elif loss_function == 'depth_cross_entropy_D':
            self.loss_function = DepthWeightedCrossEntropyLossD(**kwargs)
        elif loss_function == 'depth_cross_entropy_E':
            self.loss_function = DepthWeightedCrossEntropyLossE(**kwargs)
        elif loss_function == 'depth_cross_entropy_F':
            self.loss_function = DepthWeightedCrossEntropyLossF(**kwargs)
        else:
            print(f"ERROR: {loss_function} loss not implemented.")
            raise ValueError

        if val_loss_function == 'cross_entropy':
            self.val_loss_function = CrossEntropyLoss(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_A':
            self.val_loss_function = DepthWeightedCrossEntropyLossA(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_B':
            self.val_loss_function = DepthWeightedCrossEntropyLossB(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_C':
            self.val_loss_function = DepthWeightedCrossEntropyLossC(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_D':
            self.val_loss_function = DepthWeightedCrossEntropyLossD(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_E':
            self.val_loss_function = DepthWeightedCrossEntropyLossE(**kwargs)
        elif val_loss_function == 'depth_cross_entropy_F':
            self.val_loss_function = DepthWeightedCrossEntropyLossF(**kwargs)
        else:
            print(f"ERROR: {val_loss_function} loss not implemented.")
            raise ValueError

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
        self.compute_and_log_loss(logits, target, self.loss_function, prefix='validation/train_')
        loss = self.compute_and_log_loss(logits, target, self.val_loss_function, prefix='validation/val_')

        # log allocated memory
        self.log('mem_alloc', torch.cuda.max_memory_allocated() / 1024**2, sync_dist=True)

        return loss

    def compute_and_log_loss(self, logits, target, loss_fx, prefix=""):
        """ Compute mean loss and per layer loss and log it. Return mean loss. """
        tgt_val, tgt_dep, tgt_pos = target

        # limit target tokens, if we had to limit input size
        tgt_val = tgt_val[:, :logits.shape[1]]
        tgt_dep = tgt_dep[:, :logits.shape[1]]
        tgt_pos = tgt_pos[:, :, :logits.shape[1]]

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
            self.log(prefix + f'loss_layer_{i}', layer_loss, sync_dist=True, reduce_fx=nanmean)
        mean_loss_per_layer = nanmean(torch.tensor(loss_per_layer, device=loss.device))
        self.log(prefix + 'loss_layer_mean', mean_loss_per_layer, sync_dist=True, reduce_fx=nanmean)

        return mean_loss
