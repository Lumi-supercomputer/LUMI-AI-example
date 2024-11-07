#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=1:00:00

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/lmdb/visualtransformer-lmdb.py'
# srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/hdf5/visualtransformer-hdf5.py'
# srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/tfrecords/visualtransformer-tfrecords.py'
srun singularity exec -B data-formats/squashfs/train.squashfs:/train_images:image-src=/,data-formats/squashfs/val.squashfs:/val_images:image-src=/ $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/squashfs/visualtransformer-squashfs.py'
# srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/tar/visualtransformer-tar.py'
