import os
from ray import tune
from executable import train

# hack for Ray[Tune] + SLURM: https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
os.environ["SLURM_JOB_NAME"] = "bash"

# Define a search space
config = {
    "parameter_search": True,
    "config": "/clusterstorage/gkobsik/shape-transformer/configs/shapenet_debug.yml",
    "pretrained": None,
    "num_layers": tune.choice([2, 3, 4]),
    "embed_dim": tune.choice([4, 8, 16]),
    "num_heads": 2,
}

# Execute the hyperparameter search
analysis = tune.run(
    tune.with_parameters(train),
    resources_per_trial={
        "cpu": 6,
        "gpu": 1
    },
    config=config,
    num_samples=10,
)

print("DONE.")
