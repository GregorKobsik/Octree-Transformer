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
    "dataset": "shapenet",
    "subclass": "basket",
    "tree_depth": 7,  # allow for longer sequences up to 33k tokens
    "num_layers": tune.grid_search([4, 8, 16]),
    "embed_dim": tune.grid_search([32, 64, 128]),
    "num_heads": tune.grid_search([4, 8, 16]),
    "num_positions": tune.grid_search([1024, 2048, 4096, 8192, 16384, 32768]),
    #"attention": tune.grid_search(["basic", "fast_linear", "fast_local", "fast_reformer", "fast_favor"]),
    "attention": tune.grid_search(["reformer", "performer", "sinkhorn", "routing", "linear"]),
    "epochs": 5,
    "gpus": 1,
    "precision": 32,
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
