import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models import ShapeTransformer
from utils.data import dataloaders


def train(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    name = f"{config['name']}"

    if args.pretrained is not None:
        model = ShapeTransformer.load_from_checkpoint(args.pretrained)
        model.learning_rate = config['learning_rate']
    else:
        model = ShapeTransformer(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_positions=config['num_positions'],
            num_vocab=config['num_vocab'],
            learning_rate=config['learning_rate'],
            steps=config['steps'],
            warmup_steps=config['warmup_steps'],
        )

    train_dl, valid_dl, _ = dataloaders(config['dataset'], config['batch_size'])
    logger = pl_loggers.TensorBoardLogger("logs", name=name)

    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True)
    trainer = pl.Trainer(
        max_steps=config['steps'],
        gpus=config['gpus'],
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        checkpoint_callback=checkpoint,
        logger=logger,
    )

    trainer.fit(model, train_dl, valid_dl)
