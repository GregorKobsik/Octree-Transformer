import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import multiprocessing as mp

from utils.quadtree_MNIST import QuadtreeMNIST

# Defines a dictionary of available datasets, which can be selected.
DATASETS = {
    "mnist": QuadtreeMNIST,
}


def pad_collate(batch):
    """ Pads input sequence in each batch individually.

    Returns:
        xx_pad: padded input sequences
        yy: targets
    """
    (xx, yy) = zip(*batch)
    # TODO: select a more generic padding value, e.g. `<pad>` if possible
    xx_pad = pad_sequence(xx, batch_first=False, padding_value=2)
    return xx_pad, torch.tensor(yy)


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

    train_dl = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_cpus, pin_memory=True, collate_fn=pad_collate
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, num_workers=num_cpus, pin_memory=True, collate_fn=pad_collate
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_cpus, pin_memory=True, collate_fn=pad_collate)
    return train_dl, valid_dl, test_dl
