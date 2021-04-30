import os
import pytorch_lightning as pl

from utils.data import dataloaders
from modules import ShapeTransformer


def test(config):
    model = ShapeTransformer.load_from_checkpoint(os.path.join(config["datadir"], config["checkpoint"]))
    hparams = model.hparams

    trainer = pl.Trainer(gpus=hparams.gpus)
    _, _, test_dl = dataloaders(hparams.dataset, hparams.batch_size)

    trainer.test(model, test_dataloaders=test_dl)
