import torch
from torch.utils.data import DataLoader, random_split

import multiprocessing as mp

from data import (
    QuadtreeMNIST,
    OctreeShapeNet,
)
from data.padding import BasicPadding
from factories import create_data_transform

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
    embedding='basic',
    architecture="encoder_decoder",
    datapath="datasets",
):
    """ Loads datasets for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        embedding: Defines the used token embedding of the shape transformer.
        architecture: Defines whether the transformer uses a 'encoder_only' or 'encoder_decocer' architecture.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_ds: Dataset with training data.
        valid_ds: Dataset with validation data.
        test_ds: Dataset with test data.

    """
    # select data transform function
    transform_fn = create_data_transform(embedding, architecture, spatial_dim[dataset])

    # initialize arguments
    kwargs = {
        "root": datapath,
        "download": True,
        "num_workers": mp.cpu_count(),
        "subclass": subclass,
        "resolution": resolution,
        "mode": 'iterative' if architecture == "encoder_decoder" else 'successive',
        "transform": transform_fn,
    }

    # load train and test datasets
    train_ds = DATASETS[dataset](train=True, **kwargs)
    test_ds = DATASETS[dataset](train=False, **kwargs)

    # 90/10 splitt for training and validation data
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
    embedding,
    architecture,
    batch_size,
    datapath="datasets",
):
    """ Creates dataloaders for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        embedding: Defines the used token embedding of the shape transformer.
        architecture: Defines whether the transformer uses a 'encoder_only' or 'encoder_decocer' architecture.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_dl: Dataloader with training data.
        valid_dl: Dataloader with validation data.
        test_dl: Dataloader with test data.
    """

    # load datasets
    train_ds, valid_ds, test_ds = datasets(dataset, subclass, resolution, embedding, architecture, datapath)

    # select padding function
    pad_collate = BasicPadding(architecture)

    # initialize arguments
    kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "collate_fn": pad_collate,
        "num_workers": mp.cpu_count(),
    }

    # create dataloaders
    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_dl = DataLoader(valid_ds, **kwargs)
    test_dl = DataLoader(test_ds, **kwargs)

    return train_dl, valid_dl, test_dl
