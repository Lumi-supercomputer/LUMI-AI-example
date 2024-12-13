#!/bin/bash
#SBATCH --job-name=comp-seq
#SBATCH --output=comp-seq-%j.out
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source ../../venv-extension/bin/activate && python compare-dataset-tiny.py -n 1 -ff "hdf5" -N 100000'