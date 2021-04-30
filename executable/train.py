import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from modules import ShapeTransformer
from utils.data import dataloaders

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    GPUStatsMonitor,
)
from callbacks import (
    TensorboardImageSampler,
    WeightsAndBiasesLogger,
    TrackedGradientOutput,
)

from ray.tune.integration.pytorch_lightning import TuneReportCallback


def compute_train_steps(train_dl, epochs, accumulate_grad_batches=1, n_gpus=1, n_nodes=1):
    total_devices = n_gpus * n_nodes
    train_batches = len(train_dl) // total_devices
    return (epochs * train_batches) // accumulate_grad_batches


def train(config):
    # load pre-configuration file
    with open(config["config"], "rb") as f:
        pre_config = yaml.safe_load(f)
    # supplement missing keys from pre-configuration
    for c in pre_config:
        if not config.get(c):
            config[c] = pre_config[c]

    # load data
    train_dl, valid_dl, _ = dataloaders(config['dataset'], config['subclass'], config['batch_size'])

    # setup tensorboard logging
    logger = pl_loggers.TensorBoardLogger("logs", name=config['name'])
    callbacks = [ModelCheckpoint(
        filename="best",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )]
    if config['log_gradient']:
        callbacks.append(TrackedGradientOutput(global_only=True))
    if config['log_weights_and_biases']:
        callbacks.append(WeightsAndBiasesLogger(log_every_n_epoch=1))
    if config['log_gpu'] == 'full':
        callbacks.append(GPUStatsMonitor(intra_step_time=True, inter_step_time=True))
    if config['log_learning_rate']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config['sample_images']:
        callbacks.append(
            TensorboardImageSampler(
                dataset=valid_dl.dataset,
                num_examples=3,
                num_samples=3,
                log_every_n_epoch=1,
            )
        )
    if config['parameter_search']:
        callbacks.append(TuneReportCallback({"loss": "val_loss"}, on="validation_end"))

    pl.seed_everything(seed=None)
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        gpus=config['gpus'],
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger,
        track_grad_norm=2 if config['log_gradient'] else -1,
        accelerator='ddp' if config['gpus'] > 1 else None,
        gradient_clip_val=1.0,
        log_gpu_memory='min_max' if config['log_gpu'] else None,
        weights_summary='full',
    )

    train_steps = compute_train_steps(
        train_dl,
        config['epochs'],
        config['accumulate_grad_batches'],
        config['gpus'],
    )

    if config["pretrained"] is not None:
        model = ShapeTransformer.load_from_checkpoint(config["pretrained"])
    else:
        model = ShapeTransformer(train_steps=train_steps, **config)

    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

    trainer.fit(model, train_dl, valid_dl)
