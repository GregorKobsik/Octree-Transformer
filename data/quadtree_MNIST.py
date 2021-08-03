import numpy as np

from typing import Any, Callable
from torchvision import datasets


class QuadtreeMNIST(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        subclass: str = None,
        resolution: int = 32,  # TODO: allow custom resolution
        transform: Callable = None,
        **kwargs,
    ) -> None:
        super(QuadtreeMNIST, self).__init__(root, train=train, download=download, transform=transform)
        """ Initializes the basic MNIST dataset.

        Args:
            root: Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train: Defines whether to load the train or test dataset.
            download: If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            subclass: Unused - needed for consistent API with other downloadable datasets.
            resolution: Unused - needed for consistent API with other downloadable datasets.
            transform: Holds a transform module, which can be used for data augmentation.
        """
        if resolution != 32:
            print("WARNING: Currently only a resolution of 32 is available. Continue with resolution of 32.")
        self.resolution = 32

    def __getitem__(self, index: int) -> Any:
        """ Returns a single sample from the dataset. """
        img = self.data[index]

        # pad image to (32,32) and binarize with threshold (0.1)
        pixels = np.pad(img, (2, 2)) > 0.1

        if self.transform is not None:
            return self.transform(pixels)
        else:
            return pixels
