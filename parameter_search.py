import os
from ray import tune
from executable import train

# hack for Ray[Tune] + SLURM: https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
os.environ["SLURM_JOB_NAME"] = "bash"

# Define a search space
config = {
    "parameter_search": True,
    "config": "/clusterstorage/gkobsik/shape-transformer/configs/shapenet_debug.yml",
    "datapath": "/clusterstorage/gkobsik/shape-transformer/data",
    "pretrained": None,
    "num_layers": tune.grid_search([2, 4, 8]),
    "embed_dim": tune.grid_search([32, 64, 128]),
    "num_heads": tune.grid_search([4, 8, 16]),
    "num_positions": tune.grid_search([1024, 2048, 4096, 8192, 16384, 32768]),
    "attention": tune.grid_search(["basic", "performer", "fast_linear", "fast_local", "fast_reformer", "fast_favor"]),
    "epochs": 2,
    "gpus": 1,
    "precision": 32,
}

# Execute the hyperparameter search
analysis = tune.run(
    tune.with_parameters(train),
    resources_per_trial={
        "cpu": 6,
        "gpu": 1
    },
    config=config,
)

print("DONE.")
