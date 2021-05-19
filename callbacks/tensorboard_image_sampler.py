import torch
import torchvision

from pytorch_lightning import Callback
from utils.quadtree import Quadtree


class TensorboardImageSampler(Callback):
    def __init__(
        self,
        dataset,
        num_samples=3,
        num_examples=3,
        input_depth=4,
        output_depth=(4, 5, 6),
        log_every_n_epoch=5,
    ):
        super().__init__()
        self.input_values = []
        for i in range(num_examples):
            value, _, pos_x, pos_y, _ = dataset[i]
            self.input_values += [value.numpy()]
        self.resolution = (pos_x[0].numpy(), pos_y[0].numpy())

        self.num_samples = num_samples
        self.num_examples = num_examples
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.log_every_n_epoch = log_every_n_epoch
        assert self.log_every_n_epoch != 0

    def sample(self, trainer, pl_module):
        hparams = pl_module.hparams

        for i, val in enumerate(self.input_values):
            # discard some depth layers (down-scaling)
            qtree = Quadtree().insert_sequence(val, self.resolution)
            value, depth, pos_x, pos_y = qtree.get_sequence(depth=self.input_depth, return_depth=True, return_pos=True)

            # transform sequences to tensors and push to correct device
            value = torch.tensor(value, device=pl_module.device).long()
            depth = torch.tensor(depth, device=pl_module.device).long()
            pos = torch.tensor([pos_x, pos_y], device=pl_module.device).long()

            images = torch.tensor([], device=pl_module.device)

            for _ in range(self.num_samples):
                # sample sequence / predict shape based on input (super-resolution)
                # predicted_value = sample_sequence(
                #     pl_module,
                #     value,
                #     depth,
                #     pos,
                #     2,
                #     hparams.num_positions,
                #     hparams.tree_depth,
                # ).cpu().numpy()

                predicted_value = []

                # reconstuct images from sequence
                qtree_pred = Quadtree().insert_sequence(
                    predicted_value,
                    self.resolution,
                    autorepair_errors=True,
                    silent=True,
                )
                images = torch.cat(
                    [
                        images,
                        torch.tensor([qtree.get_image(hparams.tree_depth, mode='color')], device=pl_module.device),
                        torch.tensor(
                            [qtree_pred.get_image(d, mode='color') for d in self.output_depth], device=pl_module.device
                        ),
                    ]
                )

            # create a grid with images and log them to tensorboard
            grid = torchvision.utils.make_grid(
                tensor=images.unsqueeze(1),
                nrow=len(self.output_depth) + 1,
                pad_value=1.0,
            )
            str_title = f"{pl_module.__class__.__name__}_sample_{i}"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch == 0 and self.log_every_n_epoch > 0:
            self.sample(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch != 0 and self.log_every_n_epoch > 0:
            self.sample(trainer, pl_module)
