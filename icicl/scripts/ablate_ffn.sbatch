#!/bin/bash
# Job name:
#SBATCH --job-name=ablate_ffn
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=gpu-a100-dev
#
# Job progress file
#SBATCH --output=ablate_ffn.out
#
# Error file:
#SBATCH --error=ablate_ffn.err      
#
# Request one node:
#SBATCH --nodes=1
# 
# Use all available memory
#SBATCH --mem=0  
#
# number of tasks
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit: HH:MM:SS 12 hr
#SBATCH --time=01:00:00
#
uv run icicl/train_models_noffn.py

