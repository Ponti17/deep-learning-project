#!/bin/bash
#SBATCH --job-name=helloMnist
#SBATCH --account=project_465001543
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

srun singularity exec -B /scratch/project_465001543/:/data ../hoverImage.sif python ../train_puma.py --configs configs/config.yml