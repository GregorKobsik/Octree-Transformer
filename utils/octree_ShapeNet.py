import os
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from typing import Tuple, Any
from tqdm.contrib.concurrent import process_map

from utils.octree import Octree
from utils.hsp_loader import load_hsp

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
    subfolders = ["value", "depth", "pos_x", "pos_y", "pos_z", "target"]
    type_folders = ['iterative', 'successive']

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        num_workers: int = None,
        subclass: str = "all",
        resolution: int = 32,
        iterative: bool = False,
        **kwargs,
    ) -> None:
        """ Initializes the voxelized ShapeNet dataset and performs a Octree transformation afterwards. """
        self.root = root
        self.train = train  # training set or test set
        self.num_workers = num_workers
        self.type_folder = self.type_folders[0] if iterative else self.type_folders[1]
        self.class_folder = _class_folder_map[subclass]
        self.resolution = resolution

        # check if data already exists, otherwise create it accordingly
        self.octree_transform()

        data_file = self.training_file if self.train else self.training_file  # TODO: add train-test splitt

        self.value = torch.load(os.path.join(self.resolution_folder, self.subfolders[0], data_file))
        self.depth = torch.load(os.path.join(self.resolution_folder, self.subfolders[1], data_file))
        self.pos_x = torch.load(os.path.join(self.resolution_folder, self.subfolders[2], data_file))
        self.pos_y = torch.load(os.path.join(self.resolution_folder, self.subfolders[3], data_file))
        self.pos_z = torch.load(os.path.join(self.resolution_folder, self.subfolders[4], data_file))
        self.target = torch.load(
            os.path.join(self.resolution_folder, self.subfolders[5], data_file)
        ) if iterative else self.value

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
            torch.tensor(np.stack([self.pos_x[index], self.pos_y[index], self.pos_z[index]], axis=-1)),
            torch.tensor(self.target[index]),
        )

    def __len__(self) -> int:
        return len(self.value)

    @property
    def dataset_folder(self) -> str:
        return os.path.join('/clusterarchive/ShapeNet/voxelization')

    @property
    def octree_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def resolution_folder(self) -> str:
        return os.path.join(self.octree_folder, self.type_folder, self.class_folder, str(self.resolution))

    def _check_exists_octree(self) -> bool:
        return np.all(
            [
                # TODO: add train-test splitt
                # os.path.exists(os.path.join(dir, subfolder, self.test_file)) and
                os.path.exists(os.path.join(self.resolution_folder, subfolder, self.training_file))
                for subfolder in self.subfolders
            ]
        )

    def _transform_voxels(self, data_path):
        voxels = load_hsp(data_path, self.resolution)
        otree = Octree().insert_voxels(voxels)
        sequence = otree.get_sequence(return_depth=True, return_pos=True)
        return (*sequence, sequence[0])

    def octree_transform(self) -> None:
        """Transform the ShapeNet data if it doesn't exist in octree_folder already."""
        if self._check_exists_octree():
            return

        for subfolder in self.subfolders:
            os.makedirs(os.path.join(self.resolution_folder, subfolder), exist_ok=True)

        print('Transforming... this might take some minutes.')

        subdir = "*" if self.class_folder == "all" else self.class_folder
        data_paths = glob(self.dataset_folder + '/' + subdir + '/*.mat')

        training_transformed = np.asarray(
            process_map(self._transform_voxels, data_paths, max_workers=self.num_workers, chunksize=1), dtype=object
        )

        for i, subfolder in enumerate(self.subfolders):
            with open(os.path.join(self.resolution_folder, subfolder, self.training_file), 'wb') as f:
                torch.save(training_transformed[:, i], f)

        print('Done!')
