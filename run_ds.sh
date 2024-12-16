#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=0:10:00

n_gpus_per_node=8
n_nodes=1

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529	
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# export CXX for compiling PyTorch extensions in container
srun singularity exec $CONTAINER bash -c 'export CXX=g++-12; $WITH_CONDA && source myenv_post_upgrade2/bin/activate && torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json'

