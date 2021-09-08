#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --job-name=deep_steg

module load python/3.8.2
module load scipy-stack
source env/bin/activate

python deep_steg.py

