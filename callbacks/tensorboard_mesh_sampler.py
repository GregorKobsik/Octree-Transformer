import torch

from pytorch_lightning import Callback
from utils.octree import Octree
from utils.sample import sample_sequence
from simple_3dviz import Mesh


class TensorboardImageSampler(Callback):
    def __init__(
        self,
        dataset,
        num_examples=1,
        input_depth=4,
        output_depth=6,
        log_every_n_epoch=5,
    ):
        super().__init__()
        self.input_sequences = []
        for i in range(num_examples):
            seq, _, pos_x, _, _ = dataset[i]
            self.input_sequences += [seq.numpy()]
        self.resolution = pos_x[0].numpy()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.log_every_n_epoch = log_every_n_epoch
        assert self.log_every_n_epoch != 0

    def sample(self, trainer, pl_module):
        hparams = pl_module.hparams

        for i, seq in enumerate(self.input_sequences):
            # discard some depth layers (down-scaling)
            octree = Octree().insert_sequence(seq, self.resolution)
            seq, depth, pos_x, pos_y, pos_z = octree.get_sequence(
                depth=self.input_depth, return_depth=True, return_pos=True
            )

            # transform sequences to tensors and push to correct device
            seq = torch.tensor(seq, device=pl_module.device).long()
            depth = torch.tensor(depth, device=pl_module.device).long()
            pos = torch.tensor([pos_x, pos_y, pos_z], device=pl_module.device).long()

            voxels = torch.tensor([], device=pl_module.device)

            # sample sequence / predict shape based on input (super-resolution)
            predicted_seq = sample_sequence(
                pl_module,
                seq,
                depth,
                pos,
                3,
                hparams.num_positions,
                hparams.tree_depth,
            ).cpu().numpy()

            # reconstuct voxels from sequence
            octree_pred = Octree().insert_sequence(
                predicted_seq,
                self.resolution,
                autorepair_errors=True,
                silent=True,
            )
            voxels = torch.tensor([octree_pred.get_voxels(hparams.tree_depth, mode='color')], device=pl_module.device)

            # Build a voxel grid from the voxels
            mesh = Mesh.from_voxel_grid(voxels=voxels != 0, sizes=(1, 1, 1))

            # Log the vertices to tensorboard
            str_title = f"{pl_module.__class__.__name__}_sample_{i}"
            trainer.logger.experiment.add_mesh(str_title, vertices=mesh._vertices, global_step=trainer.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch == 0 and self.log_every_n_epoch > 0:
            self.sample(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epoch != 0 and self.log_every_n_epoch > 0:
            self.sample(trainer, pl_module)
