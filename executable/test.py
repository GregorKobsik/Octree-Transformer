import yaml

import pytorch_lightning as pl

from utils.data import dataloaders
from models import ShapeTransformer


def test(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    trainer = pl.Trainer(gpus=config['gpus'])
    _, _, test_dl = dataloaders(config['dataset'], config['batch_size'])
    model = ShapeTransformer.load_from_checkpoint(args.checkpoint)
    trainer.test(model, test_dataloaders=test_dl)
