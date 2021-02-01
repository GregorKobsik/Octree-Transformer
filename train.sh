#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-cluster
#SBATCH --output=out/train.out
#SBATCH --time=12:00:00

HOME=/clusterstorage/gkobsik
source $HOME/.bashrc

conda activate research
python src/main.py train configs/mnist_xxs.yml
# tail -f out/train.out
