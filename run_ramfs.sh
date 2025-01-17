#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2:00:00
#SBATCH --output=single_GPU.out.2h

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

srun singularity exec $CONTAINER bash -c '
  time cp -a train_images.hdf5 /tmp/. ;
  $WITH_CONDA && source myenv_post_upgrade/bin/activate && time python visualtransformer_ramfs.py  ;
  time /bin/cp -a /tmp/vit_b_16_imagenet.pth ./vit_b_16_imagenet.pth.$$ ;
  time /bin/cp -a /tmp/train_images.hdf5     ./train_images.hdf5.$$'
