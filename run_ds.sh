#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=0:10:00

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=56
##SBATCH --mem=480G

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529	
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

# With torchrun:
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
# export CXX for compiling PyTorch extensions in container
srun singularity exec $CONTAINER bash -c 'export CXX=g++-12; $WITH_CONDA && source myenv_post_upgrade2/bin/activate && torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json'

# # # Without torchrun:
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
# export WORLD_SIZE=$SLURM_NPROCS
# export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# # Set up the CPU bind masks (can only be used with full node runs (standard-g or small-g with slurm argument `--exclusive`))
# CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec $CONTAINER bash -c "export CXX=g++-12; export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID \
#                                                                                 $WITH_CONDA && source myenv_post_upgrade2/bin/activate && \
#                                                                                 time python ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json"
