#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=11G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-cluster
#SBATCH --output=out/train.out
#SBATCH --time=12:00:00

HOME=/clusterstorage/gkobsik
source $HOME/.bashrc

conda activate research
python main.py train configs/mnist_xs.yml
# tail -f out/train.out
