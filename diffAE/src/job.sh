#!/bin/bash
#
#SBATCH --job-name=diff_ae
#SBATCH --partition=common
#SBATCH --qos=1gpu4h
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --output=diff_ae.txt
#SBATCH --export=ALL,CUDA_LAUNCH_BLOCKING=1

pip3 install -r requirements.txt
python3 train.py hydra.job.chdir=True