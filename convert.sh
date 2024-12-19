#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=small
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=1:00:00

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

mkdir -p data-formats/hdf5/
mkdir -p data-formats/lmdb/

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/hdf5/convert_to_hdf5.py' &
srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/lmdb/convert_to_lmdb.py'
