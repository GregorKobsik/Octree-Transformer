import numpy as np
import os

from typing import Any, Tuple
from tqdm.contrib.concurrent import process_map

import torch
from torchvision import datasets

from utils.quadtree import Quadtree


class QuadtreeMNIST(datasets.MNIST):
    subfolders = ["value", "depth", "pos_x", "pos_y", "target"]

    def __init__(self, root: str, train: bool = True, download: bool = False, num_workers: int = None) -> None:
        super(QuadtreeMNIST, self).__init__(root, train=train, download=download)
        """ Initializes the basic MNIST dataset and performs a Quadtree transformation afterwards. """
        self.num_workers = num_workers
        self.quadtree_transform()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.value = torch.load(os.path.join(self.quadtree_folder, self.subfolders[0], data_file))
        self.depth = torch.load(os.path.join(self.quadtree_folder, self.subfolders[1], data_file))
        self.pos_x = torch.load(os.path.join(self.quadtree_folder, self.subfolders[2], data_file))
        self.pos_y = torch.load(os.path.join(self.quadtree_folder, self.subfolders[3], data_file))
        self.targets = torch.load(os.path.join(self.quadtree_folder, self.subfolders[4], data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (value, depth, position x-axis, position y-axis, target)
                    where target is index of the target class.
        """
        return (
            torch.tensor(self.value[index]),
            torch.tensor(self.depth[index]),
            torch.tensor(self.pos_x[index]),
            torch.tensor(self.pos_y[index]),
            int(self.targets[index]),
        )

    def __len__(self) -> int:
        return len(self.value)

    @property
    def quadtree_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'quadtree')

    def _check_exists_quadtree(self) -> bool:
        return np.all(
            [
                os.path.exists(os.path.join(self.quadtree_folder, subfolder, self.training_file)) and
                os.path.exists(os.path.join(self.quadtree_folder, subfolder, self.test_file))
                for subfolder in self.subfolders
            ]
        )

    def _transform_img(self, img):
        img = np.pad(img, (2, 2)) > 0.1
        qtree = Quadtree().insert_image(img)
        return qtree.get_sequence(return_depth=True, return_pos=True)

    def quadtree_transform(self) -> None:
        """Transform the MNIST data if it doesn't exist in quadtree_folder already."""

        if self._check_exists_quadtree():
            return

        for subfolder in self.subfolders:
            os.makedirs(os.path.join(self.quadtree_folder, subfolder), exist_ok=True)

        training_data, training_targets = torch.load(os.path.join(self.processed_folder, self.training_file))
        test_data, test_targets = torch.load(os.path.join(self.processed_folder, self.test_file))

        print('Transforming... this might take some minutes.')

        training_transformed = np.asarray(
            process_map(self._transform_img, training_data, max_workers=self.num_workers, chunksize=10)
        )

        for i, subfolder in enumerate(self.subfolders[:4]):
            with open(os.path.join(self.quadtree_folder, subfolder, self.training_file), 'wb') as f:
                torch.save(training_transformed[:, i], f)
        with open(os.path.join(self.quadtree_folder, self.subfolders[-1], self.training_file), 'wb') as f:
            torch.save(training_targets, f)

        test_transformed = np.asarray(
            process_map(self._transform_img, test_data, max_workers=self.num_workers, chunksize=10)
        )

        for i, subfolder in enumerate(self.subfolders[:4]):
            with open(os.path.join(self.quadtree_folder, subfolder, self.test_file), 'wb') as f:
                torch.save(test_transformed[:, i], f)
        with open(os.path.join(self.quadtree_folder, self.subfolders[-1], self.test_file), 'wb') as f:
            torch.save(test_targets, f)

        print('Done!')
