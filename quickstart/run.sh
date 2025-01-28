#!/bin/bash
#SBATCH --account=project_xxxxxxxxx  # project account to bill 
#SBATCH --partition=dev-g            # other options are small-g and standard-g
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=1:00:00               # time limit

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source visualtransformer-env/bin/activate && python visualtransformer.py'
