# LUMI Machine learning guide

This guide aims to help users to migrate their machine learning applications from smaller-scale computing environments or cloud computing providers to LUMI. We provide a basic A-Z example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/). 

The Python and bash scripts used in this guide are available in this [GitHub repository](https://github.com/Lumi-supercomputer/LUMI-AI-example/tree/main).

### Requirements

This guide assumes the users have a basic understanding of machine learning and Python programming as it focuses on LUMI-specific aspects of training machine learning models. It is also assumed that the users have a user account on LUMI and are familiar with the basic usage of the LUMI supercomputer.

### Table of contents

This guide contains the following subsections:

- [Installing Python packages in a container](containers.md)
- [Storage options for training data](data_storage.md)
- [File formats for training data](file_formats.md) 
- [Scaling training to multiple GPUs and multiple nodes](multi_gpu_and_node.md)
- [Profiling and debugging machine learning applications](profiling_and_debugging.md)
- [TensorBoard visualization](visualization.md)


### Further reading

- [LUMI software library, PyTorch](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/)
- [LUMI software library, TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/)
- [LUMI software library, Jax](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/j/jax/)
- [Workshop material - Moving your AI training jobs to LUMI](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)
