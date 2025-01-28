#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source visualtransformer-env/bin/activate && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visualtransformer.py'
