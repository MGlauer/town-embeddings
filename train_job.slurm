#!/bin/bash

# Time format is D-HH:MM:SS (see "man sbatch")
# Here we allow the job to run at most ten minutes
#SBATCH --time=0-01:00:00

# Run one task on one node
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Make 48 cores available to our task (otherwise defaults to 1)
#SBATCH --cpus-per-task=8

# Use any of the compute nodes in the 'all' partition
#SBATCH --partition=all

# Redirect output and error output
#SBATCH --output=job.out
#SBATCH --error=job.err

source .venv/bin/activate
# srun yolo train data=pizzaiolo/DEMO_SAMPLE_pizzaiolo_dataset_YOLO/data.yaml model=yolo11s.pt epochs=25 lr0=0.01
srun python -m docs.test