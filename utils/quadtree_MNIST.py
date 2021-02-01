import numpy as np
import os

from typing import Any, Tuple
from tqdm.contrib.concurrent import process_map

import torch
from torchvision import datasets

from utils.quadtree import QuadTree


class QuadtreeMNIST(datasets.MNIST):
    def __init__(self, root: str, train: bool = True, download: bool = False, num_workers: int = None) -> None:
        super(QuadtreeMNIST, self).__init__(root, train=train, download=download)
        """ Initializes the basic MNIST dataset and performs a Quadtree transformation afterwards. """
        self.num_workers = num_workers
        self.quadtree_transform()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.quadtree_folder, data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sequence, target) where target is index of the target class.
        """
        return torch.tensor(self.data[index]), int(self.targets[index])

    def __len__(self) -> int:
        return len(self.data)

    @property
    def quadtree_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'quadtree')

    def _check_exists_quadtree(self) -> bool:
        return (
            os.path.exists(os.path.join(self.quadtree_folder, self.training_file)) and
            os.path.exists(os.path.join(self.quadtree_folder, self.test_file))
        )

    def _transform_img(self, img):
        img = np.pad(img, (
            2,
            2,
        ))
        qtree = QuadTree().insert_image(img, mode='binary')
        return np.array(qtree.get_sequence())

    def quadtree_transform(self) -> None:
        """Transform the MNIST data if it doesn't exist in quadtree_folder already."""

        if self._check_exists_quadtree():
            return

        os.makedirs(self.quadtree_folder, exist_ok=True)

        training_data, training_targets = torch.load(os.path.join(self.processed_folder, self.training_file))
        test_data, test_targets = torch.load(os.path.join(self.processed_folder, self.test_file))

        print('Transforming... this might take some minutes.')

        transformed_training_data = np.array(
            process_map(self._transform_img, training_data, max_workers=self.num_workers, chunksize=10)
        )
        transformed_test_data = np.array(
            process_map(self._transform_img, test_data, max_workers=self.num_workers, chunksize=10)
        )

        training_set = (transformed_training_data, training_targets)
        test_set = (transformed_test_data, test_targets)

        with open(os.path.join(self.quadtree_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.quadtree_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
