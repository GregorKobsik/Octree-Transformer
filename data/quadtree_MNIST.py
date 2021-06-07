import os
import torch
import numpy as np
import math

from typing import Any, Tuple
from tqdm.contrib.concurrent import process_map

from torchvision import datasets

from utils import kdTree


class QuadtreeMNIST(datasets.MNIST):
    subfolders = ["value", "depth", "pos"]
    type_folders = ['iterative', 'successive']

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        num_workers: int = None,
        subclass: str = None,
        resolution: int = 32,  # TODO: allow custom resolution
        iterative: bool = False,  # TODO: allow iterative setup
        **kwargs,
    ) -> None:
        super(QuadtreeMNIST, self).__init__(root, train=train, download=download)
        """ Initializes the basic MNIST dataset and performs a Quadtree transformation afterwards. """
        self.root = root
        self.train = train  # training set or test set
        self.num_workers = num_workers
        self.type_folder = self.type_folders[0] if iterative else self.type_folders[1]
        if resolution != 32:
            print("WARNING: Currently only a resolution of 32 is available. Continue with resolution of 32.")
        self.resolution = 32  # TODO: allow custom resolution
        self.iterative = iterative

        # check if data already exists, otherwise create it accordingly
        self.quadtree_transform()

        # load requested data into memory
        data_file = self.training_file if self.train else self.training_file  # TODO: add train-test splitt
        self.load_data(data_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Tuple, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (value, depth, position, target)
        """
        return (
            torch.tensor(self.value[index]),
            torch.tensor(self.depth[index]),
            torch.tensor(self.pos[index]),
            torch.tensor(self.target[index]),
        )

    def __len__(self) -> int:
        return len(self.value)

    @property
    def quadtree_path(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def resolution_path(self) -> str:
        return os.path.join(self.quadtree_path, self.type_folder, str(self.resolution))

    def _check_exists_quadtree(self) -> bool:
        return np.all(
            [
                os.path.exists(os.path.join(self.resolution_path, subfolder, self.training_file)) and
                os.path.exists(os.path.join(self.resolution_path, subfolder, self.test_file))
                for subfolder in self.subfolders
            ]
        )

    def _transform_pixels(self, img):
        # TODO: allow custom resolution
        pixels = np.pad(img, (2, 2)) > 0.1  # pad image (32,32) and binarize with threshold (0.1)
        qtree = kdTree(spatial_dim=2).insert_element_array(pixels)
        if self.iterative:
            output = []
            for i in range(2, int(math.log2(self.resolution)) + 1):
                output += [qtree.get_token_sequence(return_depth=True, return_pos=True, depth=i)]
            return output
        else:
            return qtree.get_token_sequence(return_depth=True, return_pos=True)

    def quadtree_transform(self) -> None:
        """Transform the MNIST data if it doesn't exist already."""

        if self._check_exists_quadtree():
            return

        for subfolder in self.subfolders:
            os.makedirs(os.path.join(self.resolution_path, subfolder), exist_ok=True)

        training_data, training_targets = torch.load(os.path.join(self.processed_folder, self.training_file))
        test_data, test_targets = torch.load(os.path.join(self.processed_folder, self.test_file))

        print('Transforming... this might take some minutes.')

        training_transformed = np.asarray(
            process_map(self._transform_pixels, training_data, max_workers=self.num_workers, chunksize=10)
        )
        if self.iterative:
            training_transformed = np.concatenate(training_transformed)

        for i, subfolder in enumerate(self.subfolders):
            with open(os.path.join(self.resolution_path, subfolder, self.training_file), 'wb') as f:
                torch.save(training_transformed[:, i], f)

        test_transformed = np.asarray(
            process_map(self._transform_pixels, test_data, max_workers=self.num_workers, chunksize=10)
        )
        if self.iterative:
            test_transformed = np.concatenate(test_transformed)

        for i, subfolder in enumerate(self.subfolders):
            with open(os.path.join(self.resolution_path, subfolder, self.test_file), 'wb') as f:
                torch.save(test_transformed[:, i], f)

        print('Done!')

    def load_data(self, data_file: str) -> None:
        """ Load quadtree data files into memory. """
        self.value = torch.load(os.path.join(self.resolution_path, self.subfolders[0], data_file))
        self.depth = torch.load(os.path.join(self.resolution_path, self.subfolders[1], data_file))
        self.pos = torch.load(os.path.join(self.resolution_path, self.subfolders[2], data_file))
