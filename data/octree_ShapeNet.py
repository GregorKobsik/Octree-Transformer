import os
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from typing import Tuple, Any, Callable
from tqdm.contrib.concurrent import process_map

from utils import load_hsp, kdTree

_class_folder_map = {
    "": "all",
    "full": "all",
    "all": "all",
    "airplane": "02691156",
    "bin": "02747177",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bicycle": "02834778",
    "birdhouse": "02843684",
    "boat": "02858304",
    "bookshelf": "02871439",
    "bottle": "02876657",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "camera": "02942699",
    "can": "02946921",
    "cap": "02954340",
    "car": "02958343",
    "cellphone": "02992529",
    "chair": "03001627",
    "clock": "03046257",
    "keypad": "03085013",
    "dishwasher": "03207941",
    "display": "03211117",
    "earphone": "03261776",
    "faucet": "03325088",
    "file": "03337140",
    "guitar": "03467517",
    "helmet": "03513137",
    "jar": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "loudspeaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "microwave": "03761084",
    "motorcycle": "03790512",
    "mug": "03797390",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "pot": "03991062",
    "printer": "04004475",
    "remote": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "skateboard": "04225987",
    "sofa": "04256520",
    "stove": "04330267",
    "table": "04379243",
    "telephone": "04401088",
    "tower": "04460130",
    "train": "04468005",
    "watercraft": "04530566",
    "washer": "04554684",
}


class OctreeShapeNet(Dataset):
    """ Voxelized ShapeNet Dataset. """

    training_file = 'training.pt'
    test_file = 'test.pt'
    subfolders = ["value", "depth", "pos"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        num_workers: int = None,
        subclass: str = "all",
        resolution: int = 32,
        transform: Callable = None,
        **kwargs,
    ) -> None:
        """ Initializes the voxelized ShapeNet dataset.

        Loads the data from memory or performs an octree transformation to precompute them and save to memory.

        Args:
            root: Root directory where the dataset can be found or will be saved to.
            train: Defines whether to load the train or test dataset.
            download: Unused - needed for consistent API with other downloadable datasets.
            num_workers: Defines the number of workers used to preprocess the data.
            subclass: Defines which subclass of the dataset should be loaded. Select 'all' for all subclasses.
            resolution: Defines the used resolution of the dataset.
            transform: Holds a transform module, which is used to transform raw sequences into sequence samples.
        """
        self.root = root
        self.train = train  # training or test set
        self.num_workers = num_workers
        self.class_folder = _class_folder_map[subclass]
        self.resolution = resolution
        self.transform = transform

        # check if data already exists, otherwise create it accordingly
        self.octree_transform()

        # load requested data into memory
        data_file = self.training_file if self.train else self.training_file  # TODO: add train-test splitt
        self.load_data(data_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Tuple]:
        """ Returns a single sample from the dataset. """
        value = self.value[index]
        depth = self.depth[index]
        pos = self.pos[index]

        if self.transform is not None:
            return self.transform(value, depth, pos)
        else:
            return value, depth, pos

    def __len__(self) -> int:
        return len(self.value)

    @property
    def dataset_path(self) -> str:
        return os.path.join('/clusterarchive/ShapeNet/voxelization')

    @property
    def octree_path(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def resolution_path(self) -> str:
        return os.path.join(self.octree_path, self.class_folder, str(self.resolution))

    def _check_exists_octree(self) -> bool:
        return np.all(
            [
                # TODO: add train-test splitt
                # os.path.exists(os.path.join(dir, subfolder, self.test_file)) and
                os.path.exists(os.path.join(self.resolution_path, subfolder, self.training_file))
                for subfolder in self.subfolders
            ]
        )

    def _transform_voxels(self, data_path: str):
        # TODO: catch and model resolutions below 16
        voxels = load_hsp(data_path, max(self.resolution, 16))
        octree = kdTree(spatial_dim=3).insert_element_array(voxels)
        return octree.get_token_sequence(return_depth=True, return_pos=True)

    def octree_transform(self) -> None:
        """Transform the ShapeNet data if it doesn't exist already."""

        if self._check_exists_octree():
            return

        print('Transforming... this might take some minutes.')

        # fetch paths with raw voxel data
        subdir = "*" if self.class_folder == "all" else self.class_folder
        data_paths = glob(self.dataset_path + '/' + subdir + '/*.mat')

        # transform voxels into octree representation
        training_transformed = np.asarray(
            process_map(self._transform_voxels, data_paths, max_workers=self.num_workers, chunksize=1), dtype=object
        )

        # save data
        for i, subfolder in enumerate(self.subfolders):
            os.makedirs(os.path.join(self.resolution_path, subfolder), exist_ok=True)
            with open(os.path.join(self.resolution_path, subfolder, self.training_file), 'wb') as f:
                torch.save(training_transformed[:, i], f)

        print('Done!')

    def load_data(self, data_file: str) -> None:
        """ Load octree data files into memory. """
        self.value = torch.load(os.path.join(self.resolution_path, self.subfolders[0], data_file))
        self.depth = torch.load(os.path.join(self.resolution_path, self.subfolders[1], data_file))
        self.pos = torch.load(os.path.join(self.resolution_path, self.subfolders[2], data_file))
