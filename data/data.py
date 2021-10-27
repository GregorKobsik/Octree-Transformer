import torch
from torch.utils.data import DataLoader, random_split

import multiprocessing as mp

from . import (
    QuadtreeMNIST,
    OctreeShapeNet,
)

from .transform import create_data_transform
from .collate import create_data_collate

# Defines a dictionary of available datasets, which can be selected.
DATASETS = {
    "mnist": QuadtreeMNIST,
    "shapenet": OctreeShapeNet,
}

spatial_dim = {
    "mnist": 2,
    "shapenet": 3,
}


def datasets(
    dataset,
    subclass="all",
    resolution=32,
    transform='basic',
    datapath="datasets",
):
    """ Loads datasets for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        transform: Data transformation and augmentation functions.
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_ds: Dataset with training data.
        valid_ds: Dataset with validation data.
        test_ds: Dataset with test data.

    """
    # initialize arguments
    kwargs = {
        "root": datapath,
        "download": True,
        "subclass": subclass,
        "resolution": resolution,
        "transform": transform,
    }

    # load train and test datasets
    train_ds = DATASETS[dataset](train=True, **kwargs)
    test_ds = DATASETS[dataset](train=False, **kwargs)

    # 95/5 splitt for training and validation data
    train_size = int(0.95 * len(train_ds))

    # reproducable split
    train_ds, valid_ds = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    return train_ds, valid_ds, test_ds


def dataloaders(
    dataset,
    subclass,
    resolution,
    transform,
    embedding,
    architecture,
    batch_size,
    num_workers,
    position_encoding,
    num_positions,
    datapath="datasets",
):
    """ Creates dataloaders for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        transform: Defines data transformation and augmentation functions.
        embedding: Defines the used token embedding of the shape transformer.
        architecture: Defines the architecture of the transformer.
        batch_size: Defines the batch size for the data loader
        num_workers: Defines the number of workers for the data loader
        position_encoding: Defines the positional encoding of the data.
        num_positions: Maximum length of the input token sequence after embedding.
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_dl: Dataloader with training data.
        valid_dl: Dataloader with validation data.
        test_dl: Dataloader with test data.
    """
    # select data transform function
    transform_fn = create_data_transform(
        name=transform,
        spatial_dim=spatial_dim[dataset],
        resolution=resolution,
        position_encoding=position_encoding,
        num_positions=num_positions,
        embedding=embedding,
    )

    # load datasets
    train_ds, valid_ds, test_ds = datasets(dataset, subclass, resolution, transform_fn, datapath)

    # select padding function
    collate_fn = create_data_collate(architecture, embedding, resolution)

    # initialize arguments
    kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
    }

    # create dataloaders
    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_dl = DataLoader(valid_ds, **kwargs)
    test_dl = DataLoader(test_ds, **kwargs)

    return train_dl, valid_dl, test_dl
