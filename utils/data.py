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


def datasets(dataset, datapath="data"):
    """ Loads datasets for training, validation and testing.

    Args:
        dataset: Selects the dataset. Currently only 'mnist' available.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data automatically downloaded to the specified location.

    Returns:
        train_ds: Dataset with training data.
        valid_ds: Dataset with validation data.
        test_ds: Dataset with test data.

    """
    num_cpus = mp.cpu_count()

    train_ds = DATASETS[dataset](datapath, train=True, download=True, num_workers=num_cpus)
    test_ds = DATASETS[dataset](datapath, train=False, download=True, num_workers=num_cpus)

    # 90/10 splitt for training and validation data
    train_size = int(0.95 * len(train_ds))

    # reproducable split
    train_ds, valid_ds = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    return train_ds, valid_ds, test_ds


def pad_collate(dataset):
    """ Returns a padding function based on given dataset with the following properties:

    Pads input sequence in each batch individually.

    Returns:
        seq_pad: padded input sequences
        depth_pad: padded depth layer sequences
        pos_pad: padded and stacked spatial position sequences
    """
    def pad_mnist(batch):
        seq, depth, pos_x, pos_y, _ = zip(*batch)
        seq_pad = pad_sequence(seq, batch_first=True, padding_value=0)
        depth_pad = pad_sequence(depth, batch_first=True, padding_value=0)
        pos_x_pad = pad_sequence(pos_x, batch_first=True, padding_value=0)
        pos_y_pad = pad_sequence(pos_y, batch_first=True, padding_value=0)
        pos_pad = torch.stack([pos_x_pad, pos_y_pad])
        return seq_pad, depth_pad, pos_pad

    def pad_shapenet(batch):
        seq, depth, pos_x, pos_y, pos_z = zip(*batch)
        seq_pad = pad_sequence(seq, batch_first=True, padding_value=0)
        depth_pad = pad_sequence(depth, batch_first=True, padding_value=0)
        pos_x_pad = pad_sequence(pos_x, batch_first=True, padding_value=0)
        pos_y_pad = pad_sequence(pos_y, batch_first=True, padding_value=0)
        pos_z_pad = pad_sequence(pos_z, batch_first=True, padding_value=0)
        pos_pad = torch.stack([pos_x_pad, pos_y_pad, pos_z_pad])
        return seq_pad, depth_pad, pos_pad

    if str(dataset) in ('mnist'):
        return pad_mnist
    elif str(dataset) in ('shapenet'):
        return pad_shapenet


def dataloaders(dataset, batch_size, datapath="data"):
    """ Creates dataloaders for training, validation and testing.

    Args:
        dataset: Selects the dataset. Currently only 'mnist' available.
        batch_size: Defines the batch size for the data loader
        datapath: Path to the dataset. If the dataset is not found then
            the data automatically downloaded to the specified location.

    Returns:
        train_dl: Dataloader with training data.
        valid_dl: Dataloader with validation data.
        test_dl: Dataloader with test data.

    """
    train_ds, valid_ds, test_ds = datasets(dataset, datapath)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate(dataset),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate(dataset),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=pad_collate(dataset),
    )

    return train_dl, valid_dl, test_dl
