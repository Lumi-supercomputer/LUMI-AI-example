# LUMI-AI-example
Visual transformer model in PyTorch, serving as an example of how to run AI applications on LUMI. 

We use the [`torchvision vit_b_16`](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16) model and train it with the [tiny-imagenet](https://image-net.org/download-images.php) dataset. This project is meant to provide a sandbox for testing and benchmarking AI applications on LUMI and should eventually serve as an A-Z example as part of the LUMI AI documentation.
Use [bigger models](https://pytorch.org/vision/main/models/vision_transformer.html) and larger dataset if required

## HDF5 support
The imagenet dataset consists of hundreds of thousands of single jpg files. To avoid the "many small files" problem the datasets can be transformed into a single HDF5 file with the script `turn_into_hdf5.py`. Note, that this increases the size of the data by one order of magnitude as this script does not compress the data in any form. 

## Running script on LUMI
This github repo is cloned to `/project/project_462000002/LUMI-AI-example`. Training data, validation data, and the parameters of the model are in the same directory. The used container is extended via a virtual environment, as described [here](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md), since `h5py` is not included in the container. The training and validation datasets are also uploaded to the `lumi-o:imagenet/` bucket. Anyone is welcome to work in that directory in order to minimize data storage, but please create a new branch.


## Building website

Install the needed dependencies.

```
pip install -r requirements.txt
```

### Edit with live preview

run 

```
mkdocs serve
```

This command will start a live-reloading local web server that can be accessed
in a web browser via: http://127.0.0.1:8000. The local web serve will 
automatically re-render and reload the site when you edit the documentation.


### Generate the static site

To build a self-contained directory containing the full website run:

```
mkdocs build
```

The generated files will be located in the `site/` directory.
