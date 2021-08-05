import torch
import numpy as np
import scipy.ndimage as nd
from random import random


def nanmean(v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """ based on: https://github.com/pytorch/pytorch/issues/21987#issuecomment-813859270 """

    is_nan = torch.isnan(v)
    sum_nonnan = v.nansum(*args, **kwargs)
    n_nonnan = (is_nan == 0).sum(*args, **kwargs)
    return sum_nonnan / n_nonnan


def axis_scaling(array: np.ndarray) -> nd.array:
    """Performs a linear scaling of each array axis in the range of [0.75 .. 1.25] for each axis independently.

    Args:
        array (np.ndarray): Input array containing pixels/voxels.

    Returns:
        nd.array: Scaled array of values with the same shape as input.
    """
    # get resolution and dimension of input
    res = array.shape
    ndim = array.ndim

    # create affine transformation matrix
    matrix = np.zeros((ndim + 1, ndim + 1))
    matrix[ndim][ndim] = 1

    # scaling (diagonal row)
    for i in range(ndim):
        matrix[i][i] = 0.75 + 0.5 * random()
    # translation (fix centering)
    for i in range(ndim):
        matrix[i][ndim] = (1 - matrix[i][i]) * (res[i] / 2.0)

    # inverse coordinate transformation matrix
    inv_matrix = np.linalg.inv(matrix)

    # perform affine transformation
    return nd.affine_transform(array, matrix=inv_matrix, cval=0, order=0)


def piecewise_linear_warping(array: np.ndarray, axis: int = 0, symmetric: bool = True) -> nd.array:
    """Performs a piecewise linear scaling of input array given an axis.

    The array is subdivided in 5 equal segments, of which each one is scaled in the range of [0.75 .. 1.25].False

    Args:
        array (np.ndarray): Input array containing pixels/voxels.
        axis (int, optional): Defines the axis which will be scaled. Defaults to 0.
        symmetric (bool, optional): If 'True' performs symmetrical scaling around the center, otherwise all segments
            are scaled independently. Defaults to True.

    Returns:
        nd.array: Scaled array of values with the same shape as input.
    """
    # get resolution and dimension of input
    res = array.shape[axis]
    zoom = np.ones(array.ndim)

    # divide voxels in 5 equal pieces
    array = np.array_split(array, 5, axis=axis)

    # scale each piece linearly
    for i in range(5):
        zoom[axis] = 0.75 + 0.5 * random()
        array[i] = nd.zoom(array[i], zoom=zoom, mode='nearest', order=0)
        if symmetric:
            if i == 2:
                break  # early out for reaching middle segment
            else:
                array[4 - i] = nd.zoom(array[4 - i], zoom=zoom, cval=0, order=0)

    # concat scaled array pieces
    array = np.concatenate(array, axis=axis)

    # check size of array to restore input resolution
    len_axis = array.shape[axis]
    if len_axis > res:
        # slice array to initial size
        start_idx = (len_axis - res) // 2
        return array.take(indices=range(start_idx, start_idx + res), axis=axis)
    elif len_axis < res:
        # pad with zeros to match initial size
        pad_size = res - len_axis
        npad = [(0, 0)] * array.ndim
        npad[axis] = (pad_size // 2, (pad_size + 1) // 2)
        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    else:
        return array
