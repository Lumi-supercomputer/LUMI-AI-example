# Multi-GPU and Multi-Node Training

Training Deep Learning models is a resource-intensive task. When the compute and memory resources of a single GPU no longer suffice to train your model, multi-GPU and/or multi-node solutions can be leveraged to distribute your training job over multiple GPUs or nodes. Various strategies exist to distribute Deep Learning workloads, and various frameworks exist that implement those strategies. In this section, we cover two popular methods: data-parallelism using PyTorch's Distributed Data-Parallel (DDP) module and a mix of data parallelism and model sharding using the DeepSpeed library. It covers the neccessary changes to the source code and how to launch the distributed training jobs on LUMI.

## PyTorch DDP

PyTorch DDP can be used to implement data-parallelism in your training job. Data-parallel solutions are particularly useful when your training job does fit in the memory of a single GPU, but you would like to speed up the training process, for example because your are training on a large dataset.

### Source code changes
The script in ddp_visualtransformer.py implements PyTorch DDP on the visualtransformer example. The following changes to the source code are necessary:

Initialize the distributed environment:

```import torch.distributed as dist

dist.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

Wrap the model:

DistributedDataParallel

```from torch.nn.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[local_rank])
```

Change the dataloader to use the distributed sampler:

```
from torch.utils.data.distributed import DistributedSampler

...

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=7)
```

### Running the DDP example
The jobscript to run the PyTorch DDP example on a single node with all 4 GPUs (8 GCDs) is run_ds.sh.

Note that we reserve a full node and launch a single task, with 56 cpus per task:

```
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
```


We use the torch.distributed.run launcher.

```
srun singularity exec $SIF bash -c '$WITH_CONDA && source myenv_post_upgrade/bin/activate && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visualtransformer.py'
```


## DeepSpeed

DeepSpeed implements a strategy for distributed training that mixes data parallelism with sharding of model parameters. It supports various levels of model sharding and offloading of parameters to CPU memory. DeepSpeed is particularly useful when your training job does not fit in the memory of a single GPU and you would like to scale your training job to multiple GPUs and/or nodes to leverage the increased combined memory capacity, as well as speed up the training job.

### Source code changes
The script in ds_visualtransformer.py implements DeepSpeed on the visualtransformer example. The following changes to the source code are necessary:


Parsing of command-line parameters:

```
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--local-rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()
```

Initialization of the distributed environment:
```
import deepspeed

deepspeed.init_distributed()
```

Initialization of the DeepSpeed engine:
```
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters())
```

Modifications to the training loop:
```
for images, labels in train_loader:
	images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
	optimizer.zero_grad()

	outputs = model_engine(images)
	loss = criterion(outputs, labels)

	model_engine.backward(loss)
	model_engine.step()
	running_loss += loss.item()

```

### ds_config file
The file ds_config.json contains the DeepSpeed configuration parameters.

Parameters to pay attention to:
- zero_optimization: denotes the sharding strategy used by DeepSpeed.
- fp16: use of fp16 precision


```
{
  "zero_optimization": {
    "stage": 1
  },

  "fp16": {
    "enabled": false,
  }
}
```


### Running the DeepSpeed example
The jobscript to run the DeepSpeed example on a single node with all 4 GPUs (8 GCDs) is run_ds.sh.

Note that we reserve a full node and launch a single task, with 56 cpus per task:

```
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
```


We use the torchrun launcher. Note that MASTER_ADDR and MASTER_PORT have to be passed to the torchrun launcher, as well .

```
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun singularity exec $CONTAINER bash -c 'export CXX=g++-12; $WITH_CONDA && source myenv_post_upgrade/bin/activate && torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json'
```



