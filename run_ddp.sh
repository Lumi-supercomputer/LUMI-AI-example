#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=7

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.0-python-3.12-pytorch-20240801-vllm-baaedfd.sif

srun singularity exec --bind /pfs,/scratch,/projappl,/project,/flash,/appl $CONTAINER bash -c '$WITH_CONDA && source myenv_post_upgrade/bin/activate && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visualtransformer.py'
