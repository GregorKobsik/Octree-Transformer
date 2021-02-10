import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from models import ShapeTransformer
from utils.data import datasets
from utils.quadtree import QuadTree
from utils.sample import sample_sequence


def save(img_arr, dir):
    if not isinstance(img_arr, list):
        img_arr = [img_arr]
    # stack images and transform into a PIL Image
    figure = np.stack(img_arr, axis=0)
    figure = Image.fromarray(np.squeeze(figure))
    # create directory and filename for sample
    sample_dir = os.path.join(dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    filename = "figure_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    # save figure to file
    figure.save(os.path.join(sample_dir, filename))
    print(figure.shape)
    return


def show(img_arr):
    if not isinstance(img_arr, list):
        img_arr = [img_arr]

    num_imgs = len(img_arr)
    fig = plt.figure(figsize=(num_imgs / 2 + 8, 8))

    for i in range(num_imgs):
        fig.add_subplot(1, num_imgs + 1, i + 1)
        ax = plt.imshow(img_arr[i])
        ax.set_cmap('Greys')
        ax.set_clim(0, 1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


def sample(args):
    device = torch.device('cuda')  # 'cuda' or 'cpu'

    # restore model from checkpoint
    model = ShapeTransformer.load_from_checkpoint(args.checkpoint)
    hparams = model.hparams
    model.freeze()
    model = model.model.eval().to(device)

    max_seq_len = hparams.num_positions
    max_depth = hparams.tree_depth

    # get the test dataset and suffle it
    _, _, test_ds = datasets(hparams.dataset)

    # get a random image from the dataset
    seq, _, pos_x, pos_y, _ = test_ds[0]
    resolution = (pos_x[0], pos_y[0])

    # show sourca data
    qtree = QuadTree().insert_sequence(seq.numpy(), resolution)
    # show([qtree.get_image(d, mode='color') for d in range(args.input_depth, max_depth + 1)])

    # discard some depth layers (down-scaling)
    seq, depth, pos_x, pos_y = qtree.get_sequence(args.input_depth, return_depth=True, return_pos=True)

    # transform sequences to tensors and push to correct device
    seq = torch.tensor(seq).long().to(device)
    depth = torch.tensor(depth).long().to(device)
    pos_x = torch.tensor(pos_x).long().to(device)
    pos_y = torch.tensor(pos_y).long().to(device)

    # predict shape (super-resolution)
    print("Sample one example:")
    predicted_seq = sample_sequence(model, seq, depth, pos_x, pos_y, max_seq_len, max_depth).cpu().numpy()

    print(predicted_seq)
    # show images of predicted sample at different depth layers
    qtree_pred = QuadTree().insert_sequence(predicted_seq, resolution, autorepair_errors=True, silent=True)
    show([qtree_pred.get_image(d, mode='color') for d in range(args.input_depth, max_depth + 1)])
    # save images of predicted sample at different depth layers
    # save([qtree_pred.get_image(d) for d in range(args.input_depth, max_depth + 1)], args.datadir)
