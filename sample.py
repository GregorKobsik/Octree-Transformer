import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from skimage import measure
from tqdm.auto import tqdm

from sample import ShapeSampler


def save_obj(sample, path="", file_name="chair_mesh"):
    """ Use marching cubes to obtain the surface mesh and save it to an *.obj file """
    # convert volume to mesh
    sample = np.pad(sample, ((1, 1), (1, 1), (1, 1)))
    verts, faces, normals, values = measure.marching_cubes(sample, 0)
    # scale to normalized cube [-1.0, 1.0]^3
    verts /= sample.shape
    verts -= [0.5, 0.0, 0.5]
    verts *= 2.0
    # fix .obj indexing
    faces += 1

    # Save output as obj-file
    file_path = os.path.join(path, f'{file_name}.obj')
    with open(file_path, 'w') as f:
        for item in verts:
            f.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
        for item in normals:
            f.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))
        for item in faces:
            f.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))


if __name__ == "__main__":

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="samples/sampled_shapes")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--class_label", type=int, default=None)
    args = parser.parse_args()

    # load model
    checkpoint = os.path.join(args.logdir, 'checkpoints/last.ckpt')
    sampler = ShapeSampler(checkpoint_path=checkpoint, device="cpu")

    # create path & check number of existing shapes
    path = os.path.normpath(args.outdir)
    print("Save shapes to:", path)
    os.makedirs(path, exist_ok=True)
    num_objs = len(glob(path + '*.obj'))

    if args.class_label is not None:
        cls_label = torch.tensor([args.class_label], device="cpu")
    else:
        cls_label = None

    # sample shapes and save mesh as OBJ-file (marching cubes)
    for i in tqdm(range(args.num_samples), leave=True, desc="Samples"):
        output = sampler.sample_random(args.resolution, args.temperature, cls_label)
        save_obj(output, path, f"shape_{num_objs + i}")
