#!/bin/bash

# Time format is D-HH:MM:SS (see "man sbatch")
#SBATCH --time=0-04:00:00

#SBATCH --partition=gpu-stud
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6

#SBATCH --output=job.out
#SBATCH --error=job.err

source .venv/bin/activate
srun python model.py ./pizzaiolo/DEMO_SAMPLE_pizzaiolo_dataset_YOLO/