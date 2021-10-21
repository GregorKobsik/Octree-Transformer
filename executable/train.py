import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from modules import ShapeTransformer
from data import dataloaders

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    GPUStatsMonitor,
)
from callbacks import (
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
    train_dl, valid_dl, _ = dataloaders(
        dataset=config['dataset'],
        subclass=config['subclass'],
        resolution=config['resolution'],
        transform=config['transform'],
        embedding=config['embedding'],
        architecture=config['architecture'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        datapath=config['datapath'],
        position_encoding=config['position_encoding'],
    )

    # setup tensorboard logging
    logger = pl_loggers.TensorBoardLogger("logs", name=config['name'])
    callbacks = [
        ModelCheckpoint(
            filename="best",
            monitor="validation/val_loss_layer_mean",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
    ]
    if config['log_gradient']:
        callbacks.append(TrackedGradientOutput(global_only=True))
    if config['log_weights_and_biases']:
        callbacks.append(WeightsAndBiasesLogger(log_every_n_epoch=1))
    if config['log_gpu'] == 'full':
        callbacks.append(GPUStatsMonitor(intra_step_time=True, inter_step_time=True))
    if config['log_learning_rate']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config['parameter_search']:
        callbacks.append(
            TuneReportCallback({
                "loss": "validation/loss_layer_mean",
                "max_gpu_mem": "mem_alloc"
            }, on="validation_end")
        )

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
    torch.cuda.reset_peak_memory_stats()

    trainer.fit(model, train_dl, valid_dl)
