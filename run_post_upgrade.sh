#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.0-python-3.12-pytorch-20240801-vllm-baaedfd.sif

srun singularity exec --bind /pfs,/scratch,/projappl,/project,/flash,/appl $CONTAINER bash -c '$WITH_CONDA && source myenv_post_upgrade/bin/activate && python visualtransformer.py'

