import numpy as np
from scipy.io import loadmat


def load_hsp(file_path, resolution=None):
    resolution = max(resolution, 16)
    file = loadmat(file_path)

    boundary = (file["bi"] > 2).nonzero()
    boundary = np.stack([boundary[0], boundary[1], boundary[2]])

    if resolution is not None:
        grid = np.zeros((resolution, resolution, resolution))
        step = 256 // resolution
        reach = 16 // step

        full = np.stack((file["bi"] == 2.0).nonzero()) * reach
        n_full = full.shape[-1]

        if n_full > 0:
            full = np.repeat(np.expand_dims(full, 1), reach, 1)
            full = np.add(full, np.array(range(reach))[None, :, None])
            full = np.stack(
                [np.array(np.meshgrid(full[0, :, i], full[1, :, i], full[2, :, i])) for i in range(n_full)], axis=4
            ).reshape(3, -1)

            grid[full[0], full[1], full[2]] = 1

        boundary_cells = file["b"][file["bi"][boundary[0], boundary[1], boundary[2]].astype(int) - 1]

        for l in range(reach):
            for m in range(reach):
                for n in range(reach):
                    cell = boundary_cells[:, l * step:(l + 1) * step, m * step:(m + 1) * step,
                                          n * step:(n + 1) * step].max(axis=-1).max(axis=-1).max(axis=-1)
                    grid[boundary[0] * reach + l, boundary[1] * reach + m, boundary[2] * reach + n] = cell.astype(float)
    else:
        grid = np.empty(1)

    return grid


def load_chair(file_path, resolution):
    return load_hsp(file_path + "/03001627/1a8bbf2994788e2743e99e0cae970928.mat", resolution)


def load_airplane(file_path, resolution):
    return load_hsp(file_path + "/02691156/2b2cf12a1fde287077c5f5c64222d77e.mat", resolution)
