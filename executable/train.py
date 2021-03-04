import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models import ShapeTransformer
from utils.data import dataloaders

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
<<<<<<< HEAD
    GPUStatsMonitor,
=======
    # GPUStatsMonitor,
>>>>>>> remote dir init
)
from callbacks import (
    # TensorboardImageSampler,
    # WeightsAndBiasesLogger,
    TrackedGradientOutput,
)


def compute_train_steps(train_dl, epochs, accumulate_grad_batches=1, n_gpus=1, n_nodes=1):
    total_devices = n_gpus * n_nodes
    train_batches = len(train_dl) // total_devices
    return (epochs * train_batches) // accumulate_grad_batches


def train(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    name = f"{config['name']}"

    train_dl, valid_dl, _ = dataloaders(config['dataset'], config['batch_size'])
    logger = pl_loggers.TensorBoardLogger("logs", name=name)

    gradient_output = TrackedGradientOutput(global_only=True)
    # weights_and_biases = WeightsAndBiasesLogger(log_every_n_epoch=1)
    # image_sampler = TensorboardImageSampler(
<<<<<<< HEAD
    #     dataset=valid_dl.dataset,
    #     num_examples=3,
    #     num_samples=3,
    #     log_every_n_epoch=1,
    # )
    gpu_monitor = GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
=======
    #    dataset=valid_dl.dataset,
    #    num_examples=3,
    #    num_samples=3,
    #    log_every_n_epoch=1,
    # )
    # gpu_monitor = GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
>>>>>>> remote dir init
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        filename="best",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    pl.seed_everything(seed=None)
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        gpus=config['gpus'],
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[
            checkpoint,
            gradient_output,
            lr_monitor,
            # weights_and_biases,
            # image_sampler,
<<<<<<< HEAD
            gpu_monitor,
=======
            # gpu_monitor,
>>>>>>> remote dir init
        ],
        logger=logger,
        track_grad_norm=2,
        accelerator='ddp' if config['gpus'] > 1 else None,
<<<<<<< HEAD
=======
        gradient_clip_val=1.0,
        log_gpu_memory='min_max'
>>>>>>> remote dir init
    )

    train_steps = compute_train_steps(
        train_dl,
        config['epochs'],
        config['accumulate_grad_batches'],
        config['gpus'],
    )

    if args.pretrained is not None:
        model = ShapeTransformer.load_from_checkpoint(args.pretrained)
    else:
        model = ShapeTransformer(train_steps=train_steps, **config)

    trainer.fit(model, train_dl, valid_dl)
