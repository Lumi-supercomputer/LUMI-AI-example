# Working with containers on LUMI

## Base containers

Machine learning frameworks on LUMI are served as isolated environments in a form of container images with fundametal set of Python packages. LUMI uses the [Singularity](https://docs.sylabs.io/guides/main/user-guide/)(SingularityCE) container runtime.

Containers can be seen as encapsulated images of a specific environment including all libraries and tools and most importantly python packages. Container image can be based on virtually any linux distribution targeting host architecture but it still relies on host kernel and kernel drivers. This plays significant role in the case of LUMI.

Base images are available as flat sif files stored at the shared filesystem. 

## Interacting with a containerized environment

Environment from an image can be accesed in several ways:

 - Spawning a shell instance within a container (`sinularity shell` command)

 - Executing commands within a container (`singularity exec` command)

 - Running a container (`singularity run` command)

## Sigularity bindings

Typically container images are self-consistent. This is true for LUMI only for a single-node executions. Running containers across multiple compute nodes involves necessary runtime from the host.

## Singularity and Slurm



## Installing additional packages in a container 

## Integration with the LUMI software stack

## Custom images

User can also bring own container image or convert Docker image to singularity format.
