#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --time=7:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "start"

module load GpuModules

source ./env/bin/activate


python pdg_adversarial_training_2.py