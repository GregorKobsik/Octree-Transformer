import pytorch_lightning as pl

from data import dataloaders
from modules import ShapeTransformer


def test(config):
    pl.seed_everything(seed=None)
    trainer = pl.Trainer()

    model = ShapeTransformer.load_from_checkpoint(config["checkpoint_path"])
    model.freeze()
    model.eval().cuda()

    hparams = model.hparams

    _, _, test_dl = dataloaders(
        dataset=hparams['dataset'],
        subclass=hparams['subclass'],
        resolution=hparams['resolution'],
        transform=['linear_max_res', 'check_len'],
        embedding=hparams['embedding'],
        architecture=hparams['architecture'],
        batch_size=1,
        num_workers=hparams['num_workers'],
        datapath=hparams['datapath'],
        position_encoding=hparams['position_encoding'],
        num_positions=hparams['num_positions'],
    )

    trainer.test(model, test_dataloaders=test_dl)
