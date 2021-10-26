import os
import random
import sys
from argparse import ArgumentParser
from glob import glob

import torch
from skimage import measure
from tqdm.auto import tqdm

from data.octree_ShapeNet import OctreeShapeNet
from sample import ShapeSampler


def save_obj(sample, path="", file_name="chair_mesh"):
    """ Use marching cubes to obtain the surface mesh and save it to an *.obj file """
    # convert volume to mesh
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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="samples/sampled_shapes")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--subclass", type=str, default="chair")
    parser.add_argument("--class_label", type=int, default=None)
    args = parser.parse_args()

    # load model
    checkpoint = os.path.join(args.logdir, 'checkpoints/last.ckpt')
    sampler = ShapeSampler(checkpoint_path=checkpoint, device="cuda")

    # create path & check number of existing shapes
    path = os.path.normpath(args.outdir)
    print("Save shapes to:", path)
    os.makedirs(path, exist_ok=True)
    num_objs = len(glob(path + '*.obj'))

    if num_objs > 0:
        print("ERROR: directory is not empty. Return.")
        sys.exit(1)

    # load test set
    ds_test = OctreeShapeNet(
        train=False,
        subclass=args.subclass,
        resolution=args.resolution,
    )

    if args.class_label is not None:
        cls_label = torch.tensor([args.cls_label], device="cuda")
    else:
        cls_label = None

    # sample shapes and save mesh as OBJ-file (marching cubes)
    for i in tqdm(range(args.num_samples), leave=True, desc="Samples"):
        r = random.randrange(len(ds_test))
        precon = ds_test[r]

        output = sampler.sample_preconditioned(
            precon,
            precondition_resolution=args.resolution // 8,
            target_resolution=args.resolution,
            temperature=args.temperature,
            cls=cls_label
        )
        save_obj(output, path, f"shape_{i}")
