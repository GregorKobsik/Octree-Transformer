import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import multiprocessing as mp

from utils.quadtree_MNIST import QuadtreeMNIST
from utils.octree_ShapeNet import OctreeShapeNet

# Defines a dictionary of available datasets, which can be selected.
DATASETS = {
    "mnist": QuadtreeMNIST,
    "shapenet": OctreeShapeNet,
}


def datasets(dataset, subclass="all", resolution=32, iterative=False, datapath="data"):
    """ Loads datasets for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        iterative: Select whether to prepare the data for an 'iterative' or and 'successive' approach.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_ds: Dataset with training data.
        valid_ds: Dataset with validation data.
        test_ds: Dataset with test data.

    """
    kwargs = {
        "root": datapath,
        "download": True,
        "num_workers": mp.cpu_count(),
        "subclass": subclass,
        "resolution": resolution,
        "iterative": iterative,
    }

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


def pad_collate(batch):
    """ Pads input sequence in each batch individually.

    Returns:
        value_pad: padded value sequence
        depth_pad: padded depth layer sequence
        pos_pad: padded and stacked spatial position sequences
        target_pad: padded target sequence
    """
    value, depth, pos, target = zip(*batch)
    value_pad = pad_sequence(value, batch_first=True)
    depth_pad = pad_sequence(depth, batch_first=True)
    pos_pad = pad_sequence(pos, batch_first=True)
    target_pad = pad_sequence(target, batch_first=True)
    return value_pad, depth_pad, pos_pad, target_pad


def dataloaders(dataset, subclass, resolution, iterative, batch_size, datapath="data"):
    """ Creates dataloaders for training, validation and testing.

    Args:
        dataset: Select a dataset. Currently only 'mnist' and 'shapenet' available.
        subclass: Select a subclass of a dataset, if available.
        resolution: Select the underlying resolution of the selected dataset, if available.
        iterative: Select whether to prepare the data for an 'iterative' or an 'successive' approach.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data is automatically downloaded to the specified location.

    Returns:
        train_dl: Dataloader with training data.
        valid_dl: Dataloader with validation data.
        test_dl: Dataloader with test data.

    """
    num_cpus = mp.cpu_count()

    train_ds, valid_ds, test_ds = datasets(dataset, subclass, resolution, iterative, datapath)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate,
        num_workers=num_cpus,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate,
        num_workers=num_cpus,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate,
        num_workers=num_cpus,
    )

    return train_dl, valid_dl, test_dl
