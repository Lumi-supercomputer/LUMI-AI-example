# Data Storage Options

## Executive Summary
For most distributed parallel I/O-bound tasks you want to use on LUMI the stripting feature of the parallel Lustre FS.

Keep in mind that  LUMI-P has 32 OSTs and LUMI-F has 58 OSTs.

## Intro

Machine learning frameworks generally make extensive use of parallel processing to increase efficiency of development cycle. With increased number of nodes available for computations, a single storage target might become a bottleneck for I/O, especially when few large files are to be accessed.

To that end LUMI employs a parallel filesystem [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/), which allows for storage and retrieval of files in a distributed way, using striping to distribute files across multiple storage targets. These can be then accessed from multiple computation nodes in a parallel way, and so significantly removing said bottleneck.
**Please review this doc page, as the following uses some of its terminology.**

The main available filesystems on LUMI which have high priority for computations requiring high-throughput I/O are [LUMI-P](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumip/), [LUMI-F](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumip/) and RAM FS. LUMI-O provides generally significantly poorer performance, and so is not considered here in this version of the guide. 

To account for varying system loads, the below speeds show the averages of multiple benchmark runs. They were all performed after the LUMI system upgrade of summer 2024.

If you want to validate these numbers, you can run the tests yourself, by downloading the SLURM batch script [here](./LUMI-lustre-fs-benchmarks-set.sh). You will need 128GB of available quota in your /scratch/project_X/ and 128GB in /flash/project_X/.

## Striping
To maximize throughput, you want to ensure that you access different OSTs. If your data is to be striped evenly among OSTs, you only need to specify the number of stripes you want to use (unless you intend to overstripe, this is equal to the number of OSTs for a given file.)

If you know the desired "affinity" layout for your file chunks (stripes) in your OSTs, you can specify them with `--ost` option, like: `lfs setstripe --ost 1,2,1,1,1,2,2,2,1,2  my-file.dat` .

### Overstriping
A Lustre feature known as ‘overstriping’ addresses some limitations by allowing a single file to have more than one stripe per OST. The presence of more than one stripe per OST allows the full bandwidth of a given OST to be exploited while still using one file. [This](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf) presentation discusses synthetic and application I/O performance using overstriping and implications for achieving expected performance in shared-file I/O workloads. See slide #10 for a quick graphical representation of overstriping.


## LUMI-P
LUMI-P has 32 OST (Object Storage Targets), which allows for optimum performance when stripe count is set to a 32 or 64.

| Number of parallel tasks |stripe count|stripe size| read `(GBytes/sec)` | write `(GBytes/sec)` |
|--------------------------|------------|-----------|---------------------|----------------------|
| 1                        |  2         | 2m        |                     |                      |
| 2                        |  2         | 2m        |                     |                      |
| 4                        |  2         | 2m        |                     |                      |
| 8                        |  2         | 2m        |                     |                      |

## LUMI-F
LUMI-F has 58 OST (Object Storage Targets), which allows for optimum performance when stripe count is set to a 58 or a multiple of 58 ( see [](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf)

| Number of parallel tasks |stripe count|stripe size| read `(GBytes/sec)` | write `(GBytes/sec)` |
|--------------------------|------------|-----------|---------------------|----------------------|
| 1                        |  1         | 1m        |      1.71           |       1.38           |
| 1                        |  1         | 2m        |      1.71           |       1.38           |
| 1                        |  1         | 4m        |      1.71           |       1.38           |
| 2                        |  2         | 2m        |                     |                      |
| 4                        |  4         | 2m        |                     |                      |
| 8                        |  8         | 4m        |                     |                      |
| 16                       |  16        | 4m        |       22.7          |       22.3           |
| ...                      |  ...       | 4m        |                     |                      |
| 58                       |  58        | 4m        |       33.4          |       33.1           |

## RAM FS

For tasks which can make use of a non-distributed filesystem, on a single node, RAM can be effectively utulized through the RAM FS /tmp mountpoint. Obviously, RAM FS is the fastest of the storage options, but has the serious drawback of not being distributed, but rather localized to a single node. For completeness we do include the banchmark speeds in the standard-g partition.:

Mount point: `/tmp`

| Number of parallel tasks | read `(GBytes/sec)` | write `(GBytes/sec)` |
|--------------------------|---------------------|----------------------|
| 1                        |      7.2            |         7.2          |
| 2                        |      7.2            |         7.2          |
| 4                        |      7.2            |         7.2          |
| 8                        |      7.2            |         7.2          |


## Notes

### Maximum Stripe Size

While LUMI's [Lustre documentation](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/) claims that `the maximum stripe size is 4 GB` (meaning `2**32`), attempts to set this steip size produce an error:
``` bash
lfs setstripe --stripe-size 4294967296  ./$FILENAME
# lfs setstripe: error: stripe size '4294967296' over 4GB limit: Invalid argument (22)
```

The maximum is actually 64kB lower: It is `4294901760` which is `2**32 - 2**16`
```
lfs setstripe    --stripe-size $[ 2**32 - 2**16 ]    ./$FILENAME
```


### Future reading

 - [LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf)
 - [Configuring_Lustre_File_Striping](https://wiki.lustre.org/index.php/Configuring_Lustre_File_Striping)

