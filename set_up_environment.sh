#!/bin/bash

# these modules facilitate the use of singularity containers on LUMI
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

echo "Choosing the right container for the visualtransformer example"
export SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

echo "Extending the container with a virtual environment and install h5py."
singularity exec $SIF bash -c '$WITH_CONDA && python -m venv visualtransformer-env --system-site-packages && source visualtransformer-env/bin/activate && python -m pip install h5py mlflow'

# Please have a look at the terms of access (https://www.image-net.org/download.php) before using the dataset
echo "Copying training data to working directory."
cp /appl/local/training/LUMI-AI-Guide/tiny-imagenet-dataset.hdf5 train_images.hdf5
