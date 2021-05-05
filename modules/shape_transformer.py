from argparse import ArgumentParser

import torch
from torch.optim import Adam
import pytorch_lightning as pl

from modules import BasicEncoderOnlyModule, FastEncoderOnlyModule, PerformerEncoderOnlyModule
from lr_scheduler import ConstantWithWarmup
from loss import CrossEntropyLoss, DescendantWeightedCrossEntropyLoss

from torch.nn.utils.rnn import pad_sequence


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
            if attention.startswith('basic'):
                self.model = BasicEncoderOnlyModule(**kwargs)
                self.batch_first = False
            elif attention.startswith('fast'):
                self.model = FastEncoderOnlyModule(**kwargs)
                self.batch_first = True
            elif attention.startswith('performer'):
                self.model = PerformerEncoderOnlyModule(**kwargs)
                self.batch_first = True

        # encoder decoder architectures
        elif architecture == "encoder_decoder":
            # self.max_seq_len = num_positions // 2
            # self.is_encoder_decoder = True
            # if attention.startswith('basic'):
            #     self.model = BasicEncoderDecoderModule(**kwargs)
            #     self.batch_first = False
            # elif attention.startswith('fast'):
            print("ERROR: Not implemented, yet.")
            raise ValueError

        # loss functions
        if loss_function == 'cross_entropy':
            self.loss_function = CrossEntropyLoss()
        if loss_function == 'descendant_weighted_cross_entropy':
            self.loss_function = DescendantWeightedCrossEntropyLoss(spatial_dim=spatial_dim)

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

    def forward(self, value, depth, pos):
        return self.model(value, depth, pos)

    def _unpad_sequence(self, x):
        return x[x != 0]

    def _repad_sequence(self, value, depth, pos):
        value = pad_sequence(self._unpad_sequence(value[:self.max_seq_len])).long()
        depth = pad_sequence(self._unpad_sequence(depth[:self.max_seq_len])).long()
        pos = pad_sequence(self._unpad_sequence(pos[:, :self.max_seq_len])).long()
        return value, depth, pos

    def _transpose_sequence(self, value, depth, pos):
        value = torch.transpose(value, 0, 1).contiguous()
        depth = torch.transpose(depth, 0, 1).contiguous()
        pos = torch.transpose(pos, 1, 2).contiguous()
        return value, depth, pos

    def step_encoder_decoder(self, batch, batch_idx):
        # with torch.no_grad():
        #     src_value, src_depth, src_pos = batch

        #     # splitt the input based on depth, repeat random number of times to train different depths
        #     for _ in range(torch.randint(1, self.hparams.tree_depth, (1, ))):
        #         _, max_ids = torch.max(src_depth, 1)
        #         src_value, tgt_value = torch.tensor_split(src_value, max_ids)
        #         src_depth, tgt_depth = torch.tensor_split(src_depth, max_ids)
        #         src_pos, tgt_pos = torch.tensor_split(src_pos, max_ids)

        #     # repad and delimit the sequences based on "new" batches
        #     src_value, src_depth, src_pos = self._repad_sequence(src_value, src_depth, src_pos)
        #     tgt_value, tgt_depth, tgt_pos = self._repad_sequence(tgt_value, tgt_depth, tgt_pos)

        #     # 'fast-transformers' expects, unlike 'torch', batch size first and the sequence second.
        #     if self.batch_first:
        #         src_value, src_depth, src_pos = self._transpose_sequence(src_value, src_depth, src_pos)
        #         tgt_value, tgt_depth, tgt_pos = self._transpose_sequence(tgt_value, tgt_depth, tgt_pos)

        # logits = self.model(src_value, tgt_value, src_depth, tgt_depth, src_pos, tgt_pos)
        # loss = self.loss_function(logits.view(-1, logits.size(-1)), tgt_value.view(-1))
        # return loss
        return 0

    def step_encoder_only(self, batch, batch_idx):
        with torch.no_grad():
            value, depth, pos = batch

            # input lenght delimited to 'num_positions'
            value = value[:self.max_seq_len].long()
            depth = depth[:self.max_seq_len].long()
            pos = pos[:, :self.max_seq_len].long()

            # 'fast-transformers' expects, unlike 'torch', batch size first and the sequence second.
            if self.batch_first:
                value, depth, pos = self._transpose_sequence(value, depth, pos)

        logits = self.model(value, depth, pos)
        loss = self.loss_function(logits.view(-1, logits.size(-1)), value.view(-1), depth.view(-1))
        return loss

    def step(self, batch, batch_idx):
        if self.is_encoder_decoder:
            return self.step_encoder_decoder(batch, batch_idx)
        else:
            return self.step_encoder_only(batch, batch_idx)

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
