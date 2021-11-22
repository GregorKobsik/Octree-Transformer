import os
import random

from glob import glob
from torch.utils.data import Dataset
from typing import Tuple, Any, Callable
from utils import load_hsp

_class_folder_map = {
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

_folder_id_map = {
    "02691156": 0,
    "02747177": 1,
    "02773838": 2,
    "02801938": 3,
    "02808440": 4,
    "02818832": 5,
    "02828884": 6,
    "02834778": 7,
    "02843684": 8,
    "02858304": 9,
    "02871439": 10,
    "02876657": 11,
    "02880940": 12,
    "02924116": 13,
    "02933112": 14,
    "02942699": 15,
    "02946921": 16,
    "02954340": 17,
    "02958343": 18,
    "02992529": 19,
    "03001627": 20,
    "03046257": 21,
    "03085013": 22,
    "03207941": 23,
    "03211117": 24,
    "03261776": 25,
    "03325088": 26,
    "03337140": 27,
    "03467517": 28,
    "03513137": 29,
    "03593526": 30,
    "03624134": 31,
    "03636649": 32,
    "03642806": 33,
    "03691459": 34,
    "03710193": 35,
    "03759954": 36,
    "03761084": 37,
    "03790512": 38,
    "03797390": 39,
    "03928116": 40,
    "03938244": 41,
    "03948459": 42,
    "03991062": 43,
    "04004475": 44,
    "04074963": 45,
    "04090263": 46,
    "04099429": 47,
    "04225987": 48,
    "04256520": 49,
    "04330267": 50,
    "04379243": 51,
    "04401088": 52,
    "04460130": 53,
    "04468005": 54,
    "04530566": 55,
    "04554684": 56,
}


class OctreeShapeNet(Dataset):
    """ Voxelized ShapeNet Dataset. """
    def __init__(
        self,
        root: str = None,
        train: bool = True,
        download: bool = False,
        subclass: str = "all",
        resolution: int = 32,
        transform: Callable = None,
        **kwargs,
    ) -> None:
        """ Initializes the voxelized ShapeNet dataset.

        Args:
            root: Unused - needed for consistent API with other downloadable datasets.
            train: Defines whether to load the train or test dataset.
            download: Unused - needed for consistent API with other downloadable datasets.
            subclass: Defines which subclass of the dataset should be loaded. Select 'all' for all subclasses.
            resolution: Defines the used resolution of the dataset.
            transform: Holds a transform module, which can be used for data augmentation.
        """
        self.subclass = subclass
        self.resolution = resolution

        # data transformation & augmentation
        self.transform = transform

        # load requested data paths into memory
        self.fetch_data_paths(train)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Tuple]:
        """ Returns a single sample from the dataset. """
        voxels = load_hsp(self.data_paths[index], self.resolution)
        if self.transform is None:
            return (voxels, self.cls_ids[index])

        # iterate n times to find a valid output with data augmentation
        for _ in range(100):
            for _ in range(10):
                # perform data augmentation with transform
                output = self.transform(voxels)

                # return only a valid output
                if output is not None:
                    return output + (self.cls_ids[index], )

            # after a few failed iterations get a new random sample and retry data augmentation
            index = random.randrange(len(self.data_paths))
            voxels = load_hsp(self.data_paths[index], self.resolution)

        raise ValueError("Data loader could not create a valid sample within the token limit.")

    def __len__(self) -> int:
        return len(self.data_paths)

    @property
    def dataset_path(self) -> str:
        return os.path.join('/clusterarchive/ShapeNet/voxelization')

    def _load_voxels(self, data_path: str):
        return

    def fetch_data_paths(self, train: bool) -> None:
        """ Find and store data paths of input data files. """
        # fetch paths with raw voxel data
        paths = []
        cls = []
        if self.subclass is list:
            for subclass in self.subclass:
                subdir = _class_folder_map[subclass]
                cls_paths = [sorted(glob(self.dataset_path + '/' + subdir + '/*.mat'))]
                paths += cls_paths
                cls += [[_folder_id_map[subdir]] * len(cls_paths[0])]
        elif self.subclass == "all":
            for subdir in _class_folder_map.items():
                cls_paths = [sorted(glob(self.dataset_path + '/' + subdir[1] + '/*.mat'))]
                paths += cls_paths
                cls += [[_folder_id_map[subdir[1]]] * len(cls_paths[0])]
        else:
            subdir = _class_folder_map[self.subclass]
            paths = [sorted(glob(self.dataset_path + '/' + subdir + '/*.mat'))]
            cls = [[_folder_id_map[subdir]] * len(paths[0])]

        # repeatable train-test split (80-20)
        self.data_paths = []
        self.cls_ids = []
        for i, p in enumerate(paths):
            idx = int(0.8 * len(p))
            self.data_paths += p[:idx] if train else p[idx:]
            self.cls_ids += cls[i][:idx] if train else cls[i][idx:]
