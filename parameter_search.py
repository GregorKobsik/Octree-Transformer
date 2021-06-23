import os
from ray import tune
from executable import train

# hack for Ray[Tune] + SLURM: https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
os.environ["SLURM_JOB_NAME"] = "bash"

# Define a search space
config = {
    "parameter_search": True,
    "config": "/clusterstorage/gkobsik/shape-transformer/configs/shapenet_debug.yml",
    "datapath": "/clusterstorage/gkobsik/shape-transformer/datasets",
    "pretrained": None,
    "dataset": "shapenet",
    "subclass": "chair",
    "epochs": 100,
    "gpus": 2,
    "batch_size": 2,
    "loss_function": tune.grid_search(["cross_entropy", "depth_cross_entropy_A", "depth_cross_entropy_B"]),
    "name": "weighted_cross_entropy",
}

# Execute the hyperparameter search
analysis = tune.run(
    tune.with_parameters(train),
    resources_per_trial={
        "cpu": config['gpus'] * 6,
        "gpu": config['gpus']
    },
    config=config,
)

print("DONE.")
