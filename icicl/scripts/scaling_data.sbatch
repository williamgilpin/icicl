#!/bin/bash
# Job name:
#SBATCH --job-name=scaling_data
#
# Account:
#SBATCH --account=PHY22025
#
# Pick partition:
#SBATCH --partition=gpu-a100
#
# Job progress file
#SBATCH --output=scaling_data.out
#
# Error file:
#SBATCH --error=scaling_data.err      
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
#SBATCH --time=12:00:00
#
uv run icicl/scaling_law_data_sweep.py --seed 17

