#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=1:00:00


# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source myenv_post_upgrade/bin/activate && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visualtransformer_only_training_dataset.py'
