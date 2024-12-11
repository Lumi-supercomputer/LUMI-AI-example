## Introduction
Generally, there are no one-size-fits-all file format suitable for all machine learning and artificial intelligence data. Different high-performance file formats has different strategies for increasing the read/write throughput, and these strategies might not be compatible with the format of the data (e.g. variable image sizes). As a result, this compatibility must first be determined before a optimal file format can be chosen. 

Another practical issue is the data conversion necessary to change ones data from its current file format to the desired target file format. This is primarily an issue for large datasets containing more than hundreds of thousands of small files on a parallel file system like on LUMI. Converting the data to a raw format must be avoided at all cost to preserve the integrity of the file system for all users. This issue can be circumvented in various ways, one options is to prepare the data the desired file format before is it transferred to LUMI, another option is to convert directly from the initial format (often .zip) to the target file format.

Then, after the dataset is converted to the desired file format, one also need to efficiently process the dataset in PyTorch. Different formats have different requirements here as well, where some require writing custom classes, while others or plug-and-play. Custom datasets can be built in PyTorch using the [built-in base classes](https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets). 

Finally, the actual performance of the various file formats is the final deciding factor. The performance of the different file format has been analyzed with a 5GB tiny imagenet and 157GB full imagenet. For the tiny imagenet we found near identical performance for all file formats. 
## Squashfs
Squashfs is perhaps the simplest way to get started with a PyTorch AI/ML workflow on a HPC platform. It poses no restrictions in terms of compatibility, and it requires the least amount of custom data parsers and scripts out of the formats tested here. However, it does currently require a local linux system for data conversion. It is also the least performing on large datasets.

### Data conversion
Data conversion is done using the command line command line `mksquashfs` available in various Linux package managers (`apt-get`, `dnf`, `zypper`, ...). If we have a raw data folder `ILSVRC/` for imagenet, we can convert it to the squashfs file format using 
```bash
mksquashfs ILSVRC/ imagenet.squashfs
```
Then the `.squashfs` file is ready to be transferred to LUMI. This can be done in a variety of ways as seen in the [LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/movingdata/), where for this purpose `rsync` is particularly useful with the flags `--partial` and `--progress --stats` to keep partial transfers and display detailed transfer progress stats of the large single file. 

### Running PyTorch
Running PyTorch with data stored in the `squashfs` file format is particularly simple because we are already utilizing containers which were introduced in chapter XXX. The singularity container supports [mounting](https://docs.sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html#squashfs-image-files) the `squashfs` file directly into the file system when running the container,
```bash
singularity run -B inputs.squashfs:/input-data:image-src=/ mycontainer.sif
```
 Where `inputs.squashfs` is the relative directory to the stored `.squashfs` file, `:/input-data` is the folder where the data will appear inside the container file system and finally `:image-src=/` tells singularity it should be mounted as a folder (In contrast to a single file), and the `/` describes the path inside the `.squashfs` file the mount will appear. 
 For example, if the folder `ILSVRC/` has a deep tree structure such as `ILSVRC/Data/CLS-LOC/train/...`, where I only need the training data in the `train/...` folder, the bind-mount would be
```bash
 -B scratch/project_465XXXXXX/data/imagenet.squashfs:/train_images:image-src=/Data/CLS-LOC/train/
```
Where the large squashfs data is stored in the project scratch folder scratch/project_465XXXXXX/data. We can then run PyTorch using the built-in dataset `ImageFolder` as if the dataset was stored in an ordinary folder in-side the container,
```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder('/train_images')  # Data is bind-mounted at /train_images 
```

## HDF5
Hierarchical Data Format (HDF5) is a very established high-performance file format, which interfaces well with the popular `numpy` library through its `h5py` python interface. This convenience does come at a cost of poor compatibility with irregularly shaped data, such as images with varying shapes and graph networks. 

### Data conversion
Converting a dataset into the HDF5 format can be done entirely in Python. However, in order to unpacking archive data to raw data on LUMI, one needs to first write a _parser_ which can read the archive data and stream it into the HDF5 format. This parser can be written using Python native packages such as `zipfile`, `tarfile` or the higher-level `shutil`. An example of such a parser can be seen on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/generics.py#L37) for the 157GB imagenet dataset, however this is not general purpose and needs to be customized to be suitable for different dataset. When this is done, it is quite easy to create the desired HDF5 file using the `hdf5.File().create_dataset` as illustrated on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/hdf5/convert_to_hdf5.py#L14), where the data needs to fit into a large `numpy.ndarray`-like shape like for the tiny-imagenet.

### Running PyTorch
In order to create a PyTorch `DataLoader`, we need to have a PyTorch `dataset` which fetches items and knows about the size of the data. The `dataset` can be different depending on the data, and as such needs to be custom-made to each project as well. This can be done by opening the HDF5 file using the `h5py` library and fetching items using ordinary indexing of numpy-like objects. An example is illustrated where the custom dataset class is [created here](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/hdf5/hdf5_dataset.py#L6) for the tiny-imagenet and is used in a visual transformer application on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/hdf5/visualtransformer-hdf5.py#L72) as well.

## LMDB
Lightning Memory-Mapped Database (LMDB) is a very fast file format, which like squashfs imposes no restriction on the shape of the data and thus has good compatibility. This is done using large amounts of RAM to write large cache tables for near instant memory look-up. It is not necessary to store the entire cache table in memory at once.

### Data conversion
The process to convert to LMDB is quite similar to that of HDF5. We first need to parse the archive file and then process using the Python library `lmdb`. However, as this format is more flexible and not strictly bound to conventions of `numpy` arrays, creating and processing the data is slightly more complicated. An example of this is found here on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/lmdb/convert_to_lmdb.py) for the tiny-imagenet in raw format and here on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/lmdb/convert_large_to_lmdb.py) for the large imagenet in a `.zip` file format. 

### Running PyTorch
Similar to HDF5, we require a custom built `dataset` for the LMDB file-format in order to efficiently load the data into PyTorch through the `DataLoader` framework. This can likewise be created using the `lmdb` Python library as illustrated on [Github](https://github.com/Lumi-supercomputer/LUMI-AI-example/blob/95444cb13eec48f6eb78d62f73449d859d0e8414/scripts/lmdb/lmdb_dataset.py#L10). The dataset is more complicated since we now ourselves need to encode the data to binary. 

The LMDB format has more possible performance optimization options available...

## Performance
### Synthetic Benchmark
In the synthetic benchmark, we measure how quickly samples can be loaded into Python using the PyTorch `DataLoader` for the various different file formats. The looping time is measured for both the tiny and large imagenet a number of times. Here we report the measured average and standard deviation. 
For the tiny imagenet, we loop through the entire dataset of 100.000 images a total of N=10 times for each format. Each job is executed in an independent job to ensure the RAM is emptied. The result is as follows;

|          | mean (s) | std (s) |  N  |
| :------: | :------: | :-----: | :-: |
| squashfs |  48.62   |  0.86   | 10  |
|   HDF5   |  36.51   |  2.65   | 10  |
|   LMDB   |  35.26   |  1.98   | 10  |
We find that HDF5 and LMDB have roughly equal performance, where squashfs clearly performs about 33% slower. The parameters of the `DataLoader` are as follows:
`DataLoader(data, batch_size=32, shuffle=True, num_workers=7)`
That is, the data is shuffled to be loaded in a random order, and is loaded in batches of 32 at a time. The number of workers is set equal to the number of CPUs requested in the allocation. Where on [LUMI one should maximally request 7 cores per GPU request](https://lumi-supercomputer.github.io/LUMI-training-materials/User-Updates/Update-202308/responsible-use/#core-and-memory-use-on-small-g-and-dev-g).

For the large imagenet, we loop through 200.000 images out of the 1.2 million images a total of N=3 times for the compatible formats. Identical `DataLoader` parameters and job submission strategy is employed. 

|          | mean (s) | std (s) |  N  |
| :------: | :------: | :-----: | :-: |
| squashfs | 1982.16  |  50.47  |  3  |
|   LMDB   | 1546.53  |  65.07  |  3  |
where we see the speed-up of roughly 28% for LMDB compared to squashfs. 
