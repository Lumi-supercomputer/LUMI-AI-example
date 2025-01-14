# Data Storage Options

## Executive Summary
For most distributed parallel I/O-bound tasks you want to use on LUMI the stripting feature of the parallel **Lustre FS**.

As far as **stripe count** goes, keep in mind that **LUMI-P has 32 OSTs** and **LUMI-F has 58 OSTs**.

As fas as **stripe size** goes, safe value is **2^22 (4MiB)**, unless you know that your application uses a specific data chunk size.

To configure striping with `32` stripes in `4MiB` stripe chunks all files that will be created in `/your/newly/created/data/directory`, or for an `/individual/empty/file` that you will populate, run cmd like this:
``` bash
lfs   setstripe   --stripe-count 32  --stripe-size 4m  /your/newly/created/data/directory   or  /individual/empty/file
```

## Intro

Machine learning frameworks generally make extensive use of parallel processing to shorten time to obtain results, and so to increase the efficiency of development cycle. With increased number of nodes available for computations, a single storage target might become a bottleneck for I/O, especially when large files are to be accessed.

To that end LUMI employs a parallel filesystem [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/), which allows for storage and retrieval of files in a distributed way, using striping to distribute files across multiple storage targets (OSTs). These can be then accessed from multiple computation nodes in a parallel way, and so significantly removing said bottleneck.

**Please review this [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/) doc page to familiarize yourself with the terminology.**

The main available filesystems on LUMI which have high priority for computations requiring high-throughput I/O are [LUMI-P](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumip/), [LUMI-F](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumip/) and RAM FS. In the context of massive parallel processing access, [LUMI-O](https://docs.lumi-supercomputer.eu/storage/lumio/) provides generally significantly poorer performance, and so is not considered here in this version of the guide. 

To account for varying system loads, the below speeds show the averages of multiple benchmark runs. They were all performed after the LUMI system upgrade of summer 2024.

To see how exactly these numbers were obtained or if you want to validate these numbers under current system load or configuration, you can run the tests yourself by downloading the SLURM batch script [here](./LUMI-lustre-fs-benchmarks-set.sh). You will need 128GB of available quota in your /scratch/project_X/ and 128GB in /flash/project_X/.

## Striping
To maximize I/O throughput, you want to ensure that you access different OSTs. If your data files are large enough, they are to be striped evenly among OSTs, and you only need to specify the number of stripes you want to use (unless you intend to overstripe, this is equal to the number of available OSTs for a given file.)

If you know the desired layout for your file chunks (stripes) in your OSTs, you can specify them with `--ost` option, like: `lfs setstripe --ost 1,2,1,1,1,2,2,2,1,2  my-file.dat` .


### Overstriping
An advanced Lustre feature known as ‘overstriping’ addresses some limitations by allowing a single file to have more than one stripe per OST. The presence of more than one stripe per OST allows the full bandwidth of a given OST to be exploited while still using one file. [This](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf) presentation discusses synthetic and application I/O performance using overstriping and implications for achieving expected performance in shared-file I/O workloads. See slide #10 in the said PDF for a quick graphical representation of overstriping.


## LUMI-P
LUMI-P has 32 OST (Object Storage Targets), which allows for optimum performance when stripe count is set to a 32 or 64.

| Number of parallel tasks |stripe count|stripe size| read `(GBytes/sec)` per task | read `(GBytes/sec)` total | write `(GBytes/sec)` per task | write `(GBytes/sec)` total |
|--------------------------|------------|-----------|------------------------------|---------------------------|-------------------------------|----------------------------|
| 1                        |  1         | 1m        |      0.16                    |      TBD                  |       TBD                     |       TBD                  |
| 1                        |  1         | 2m        |      0.16                    |      TBD                  |       TBD                     |       TBD                  |
| 1                        |  1         | 4m        |      0.16                    |      TBD                  |       TBD                     |       TBD                  |
| 2                        |  2         | 1g        |                              |                           |                               |                            |
| 4                        |  4         | 2g        |                              |                           |                               |                            |
| 8                        |  8         | 2g        |                              |                           |                               |                            |
| 16                       |  16        | 2g        |      0.16                    |       2.57                |       TBD                     |       TBD                  |
| ...                      |  ...       | 2g        |                              |                           |                               |                            |
| 32                       |  32        | 2g        |      TBD                     |       TBD                 |       TBD                     |       TBD                  |
| 58                       |  58        | 2g        |      TBD                     |       TBD                 |       TBD                     |       TBD                  |


## LUMI-F
LUMI-F has 58 OST (Object Storage Targets), which allows for optimum performance when stripe count is set to a 58 or a multiple of 58 ( see [](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf)

| Number of parallel tasks |stripe count|stripe size| read `(GBytes/sec)` per task | read `(GBytes/sec)` total | write `(GBytes/sec)` per task | write `(GBytes/sec)` total |
|--------------------------|------------|-----------|------------------------------|---------------------------|-------------------------------|----------------------------|
| 1                        |  1         | 1m        |      1.71                    |      1.71                 |       1.38                    |       1.38                 |
| 1                        |  1         | 2m        |      1.71                    |      1.71                 |       1.38                    |       1.38                 |
| 1                        |  1         | 4m        |      1.71                    |      1.71                 |       1.38                    |       1.38                 |
| 2                        |  2         | 2m        |                              |                           |                               |                            |
| 4                        |  4         | 2m        |                              |                           |                               |                            |
| 8                        |  8         | 4m        |                              |                           |                               |                            |
| 16                       |  16        | 4m        |      1.71                    |       22.7                |       22.3                    |       22.3                 |
| ...                      |  ...       | 4m        |                              |                           |                               |                            |
| 32                       |  32        | 4m        |      TBD                     |       TBD                 |       TBD                     |       TBD                  |
| 58                       |  58        | 4m        |      TBD                     |       TBD                 |       TBD                     |       TBD                  |




## RAM FS

For tasks which can make use of a non-distributed filesystem, on a single node, RAM can be effectively utulized through the RAM FS /tmp mountpoint. Obviously, RAM FS is the fastest of the storage options, but has the serious drawback of not being distributed, but rather localized to a single node. For completeness we do include the banchmark speeds in the standard-g partition.:

Mount point: `/tmp`

| Number of parallel tasks | read `(GBytes/sec)` | write `(GBytes/sec)` |
|--------------------------|---------------------|----------------------|
| 1                        |      7.8            |         2.9          |
| 2                        |      TBD            |         TBD          |
| 4                        |      8.0            |         2.9          |
| 8                        |      TBD            |         TBD          |


## Notes

You may use `k`, `m` or `g` suffix for `--stripe-size N{k|m|g}` argument.


### Minumum Stripe Size

Stripe size is a whole multiple of 64kB

So minimum stripe size is 1 * `2**16` = 64 kB.



### Maximum Stripe Size

While LUMI's [Lustre documentation](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/) claims that `the maximum stripe size is 4 GB` (meaning `2**32`), attempts to set this stripe size produce an error:
``` bash
lfs setstripe --stripe-size 4294967296  ./FILENAME  ; echo \$?==$?     # purportedly max size
#        lfs setstripe: error: stripe size '4294967296' over 4GB limit: Invalid argument (22)
#        $?==22
```

The maximum is actually 64kB lower: It is `4294901760` == `2**32  -  1 * 2**16`  == `2**16 * ( 2**16 - 1 )` 
```
( set -x ; lfs setstripe    --stripe-size $[ 2**32 - 2**16 ]    ./FILENAME  ; echo \$?==$?  ) # max stripe size
#        + lfs setstripe --stripe-size 4294901760 ./FILENAME
#        $?==0 == success

```


### Future reading

 - [LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf)
 - [Configuring_Lustre_File_Striping](https://wiki.lustre.org/index.php/Configuring_Lustre_File_Striping)

