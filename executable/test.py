import os
import pytorch_lightning as pl

from utils.data import dataloaders
from models import ShapeTransformer


def test(args):
    model = ShapeTransformer.load_from_checkpoint(os.path.join(args.datadir, args.checkpoint))
    hparams = model.hparams

    trainer = pl.Trainer(gpus=hparams.gpus)
    _, _, test_dl = dataloaders(hparams.dataset, hparams.batch_size)

    trainer.test(model, test_dataloaders=test_dl)
