# QuickStart

This chapter covers how to set up the environment to run the [`visualtransformer.py`](../visualtransformer.py) script on LUMI. 

First, you clone this repository to LUMI via the following command:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-example.git
```

We recommend using your `/project/` or `/scratch/` directory of your project to clone the repository as your home directory (`$HOME`) contains only 20 GB of space intended to store user configuration files and personal data.

Next, navigate to the `LUMI-AI-example` directory:

```bash
cd LUMI-AI-example
```

We now need to setup the environment if we wish to run the included python scripts. We will use one of the provided PyTorch containers that we extend with a virtual environment to install additional packages (this step will be explained in more detail in the next chapter [Setting up your own environment](containers.md)). The fastest way to achieve this is to use the provided script `set_up_environment.sh`:

```bash
./set_up_environment.sh
```

If you receive a permission denied error, you can make the script executable by running:

```bash
chmod +x set_up_environment.sh
```

After the script has finished, you should see a new directory `visualtransformer-env` in the `LUMI-AI-example` directory. This directory contains the virtual environment with all the necessary packages installed that were not already provided by the PyTorch container. In addition, the script copied the training dataset for the Visual Transformer example to the `LUMI-AI-example` directory. 
For this example, we use the [Tiny ImageNet Dataset](https://paperswithcode.com/dataset/tiny-imagenet) which is already transformed into the file system friendly hdf5 format (Chapter [File formats for training data](file_formats.md) explains in detail why this step is necessary). Please have a look at the terms of access for the ImageNet Dataset [here](https://www.image-net.org/download.php).

To run the Visual Transformer example, we need to use a batch job script. We provide a batch job script [`run.sh`](../run.sh) that you can use to run the [visualtransformer.py](../visualtransformer.py) script on a single GPU on a LUMI-G node. 
A quickstart to SLURM is provided in the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/). 

To run the provided script yourself, you need to replace the `--account` flag in line 2 of the [`run.sh`](../run.sh) script with your own project account. You can find your project account by running the command `lumi-workspaces`.

After you have replaced the `--account` flag, you can submit the job to the LUMI scheduler by running:

```bash
sbatch run.sh
```

Once the job starts running, a `slurm-<jobid>.out` file will be created in the `LUMI-AI-example` directory. This file contains the output of the job and will be updated as the job progresses. The output will show Loss and Accuracy values for each epoch, similar to the following:

```bash
Epoch 1, Loss: 4.68622251625061
Accuracy: 9.57%
Epoch 2, Loss: 4.104039922332763
Accuracy: 15.795%
Epoch 3, Loss: 3.7419378942489625
Accuracy: 19.525%
Epoch 4, Loss: 3.6926351853370667
Accuracy: 21.265%
...
```

Congratulations! You have run your first training job on LUMI. The next chapter [Setting up your own environment](containers.md) will explain in more detail how the environment was set up and how you can set up your own environment for your AI projects on LUMI.

 ### Table of contents

- [Home](index.md)
- [QuickStart](quickstart.md)
- [Setting up your own environment](containers.md)
- [File formats for training data](file_formats.md) 
- [Data Storage Options](data_storage.md)
- [Multi-GPU and Multi-Node Training](multi_gpu_and_node.md)
- [Monitoring and Profiling jobs](profiling.md)
- [TensorBoard visualization](tensorboard_visualization.md)
- [MLflow visualization](mlflow_visualization.md)
