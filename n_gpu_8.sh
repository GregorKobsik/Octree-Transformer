#!/bin/bash
#SBATCH --cpus-per-task=36
#SBATCH --mem=80G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-cluster
#SBATCH --output=out/n_gpu_8.out
#SBATCH --time=5-12:00:00

HOME=/clusterstorage/gkobsik
source $HOME/.bashrc

conda activate research
python main.py train "configs/$1.yml"
# tail -f out/train.out
