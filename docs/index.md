# LUMI Machine learning guide

This guide is designed to assist users in migrating their machine learning applications from smaller-scale computing environments or various cloud computing providers to LUMI. We will walk you through a detailed example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/).

All Python and bash scripts referenced in this guide are accessible in this [GitHub repository](https://github.com/Lumi-supercomputer/LUMI-AI-example/tree/main).

### Requirements

Before proceeding, please ensure you meet the following prerequisites:

* A basic understanding of machine learning concepts and Python programming. This guide will focus primarily on aspects specific to training models on LUMI.
* An active user account on LUMI and familiarity with its basic operations.

### Table of contents

The guide is structured into the following sections:

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
