import os
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from typing import Any, Tuple
from tqdm.contrib.concurrent import process_map

from utils.octree import Octree
from utils.hsp_loader import load_hsp


class OctreeShapeNet(Dataset):
    """ Voxelized ShapeNet Dataset. """

    training_file = 'training.pt'
    test_file = 'test.pt'
    subfolders = ["seq", "depth", "pos_x", "pos_y", "pos_z"]

    def __init__(self, root: str, train: bool = True, download: bool = False, num_workers: int = None) -> None:
        """ Initializes the voxelized ShapeNet dataset and performs a Octree transformation afterwards. """
        self.root = root
        self.train = train  # training set or test set
        self.num_workers = num_workers
        self.octree_transform()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.training_file  # TODO: add train-test splitt

        self.seq = torch.load(os.path.join(self.octree_folder, self.subfolders[0], data_file))
        self.depth = torch.load(os.path.join(self.octree_folder, self.subfolders[1], data_file))
        self.pos_x = torch.load(os.path.join(self.octree_folder, self.subfolders[2], data_file))
        self.pos_y = torch.load(os.path.join(self.octree_folder, self.subfolders[3], data_file))
        self.pos_z = torch.load(os.path.join(self.octree_folder, self.subfolders[4], data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sequence, depth, position x-axis, position y-axis, position z-axis)
        """
        return (
            torch.tensor(self.seq[index]),
            torch.tensor(self.depth[index]),
            torch.tensor(self.pos_x[index]),
            torch.tensor(self.pos_y[index]),
            torch.tensor(self.pos_z[index]),
        )

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def dataset_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def octree_folder(self) -> str:
        return os.path.join(self.dataset_folder, 'octree')

    def _check_exists_octree(self) -> bool:
        return np.all(
            [
                # TODO: add train-test splitt
                # os.path.exists(os.path.join(self.octree_folder, subfolder, self.test_file)) and
                os.path.exists(os.path.join(self.octree_folder, subfolder, self.training_file))
                for subfolder in self.subfolders
            ]
        )

    def _transform_voxels(self, data_path):
        voxels = load_hsp(data_path, 256)
        otree = Octree().insert_voxels(voxels)
        return otree.get_sequence(return_depth=True, return_pos=True)

    def octree_transform(self) -> None:
        """Transform the ShapeNet data if it doesn't exist in octree_folder already."""

        if self._check_exists_octree():
            return

        print('Transforming... this might take some minutes.')

        data_paths = glob(self.dataset_folder + '/*/*.mat')  # TODO: add train-test splitt
        training_transformed = np.asarray(
            process_map(self._transform_voxels, data_paths, max_workers=self.num_workers, chunksize=1)
        )

        for i, subfolder in enumerate(self.subfolders):
            os.makedirs(os.path.join(self.octree_folder, subfolder), exist_ok=True)
            with open(os.path.join(self.octree_folder, subfolder, self.training_file), 'wb') as f:
                torch.save(training_transformed[:, i], f)

        print('Done!')
