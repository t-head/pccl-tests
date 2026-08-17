# PCCL Tests

These tests check both the performance and the correctness of PCCL operations.

## Build

To build the tests, just type `./build.sh`.

If PPU_SDK is not installed in /usr/local/PPU_SDK, you may setup PPU_SDK environment. Similarly, if PCCL is not installed in PPU_SDK, you may specify `PCCL_HOME`.

```shell
$ source /path/to/PPU_SDK/envsetup.sh
$ PCCL_HOME=/path/to/pccl ./build.sh
```

PCCL tests rely on MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to add `--use_mpi` and set `MPI_HOME` to the path where MPI is installed.

```shell
$ source /path/to/PPU_SDK/envsetup.sh
$ MPI_HOME=/path/to/mpi PCCL_HOME=/path/to/pccl ./build.sh --use_mpi
```

## Usage

PCCL tests can run on multiple processes, multiple threads, and multiple CUDA devices per thread. The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=CUDA devices) will be equal to `(number of processes)*(number of threads)*(number of GPUs per thread)`.

### Quick examples

Run on single node with 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MiB (Mebibytes), doubling between each test (`-f 2`) :

```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Run 64 MPI processes on nodes with 8 GPUs each, for a total of 64 GPUs spread across 8 nodes.
Scanning from 8 Bytes to 8GiB (Gibibytes), doubling between each test (`-f 2`).
(NB: The pccl-tests binaries must be compiled with `MPI=1` for this case)

```shell
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

### Arguments

All tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` Number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` Number of GPUs per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` Minimum size to start with. Default : 32MiB.
  * `-e,--maxbytes <max size in bytes>` Maximum size to end at. Default : 32MiB.
  * Increments can be either fixed or a multiplication factor. Only one of those should be used.
    * `-i,--stepbytes <increment size>` Fixed increment between sizes. Default : 1MiB.
    * `-f,--stepfactor <increment factor>` Multiplication factor between sizes. Default : 1 (disabled).
    * `--sizelist <size>[,size]` Comma-separated list of sizes to test (alternative to min/max/step).
    * `--random_size_num <N>` Generate N random sizes between minbytes and maxbytes.
* PCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/mulsum/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <pccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root>` Specify which root to use. Only for operations with a root like Broadcast or Reduce. Default : 0.
* Performance
  * `-n,--iters <iteration count>` Number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` Number of warmup iterations (not timed). Default : 5.
  * `-m,--agg_iters <aggregation count>` Number of operations to aggregate together in each iteration. Default : 1.
  * `-a,--average <0/1/2/3>` Report average iteration time: 0=Rank0, 1=Avg, 2=Min, 3=Max. Default : 2 (Min).
  * `-G,--hggcgraph <num graph launches>` Number of HGCG graph launches for graph-based execution. Default : 0.
  * `--elapsed_type <0/1/2>` Timing source: 0=CPU time, 1=HGCG event time, 2=HGPTI time. Default : 0.
  * `--api_time` Report API overhead time (only for report type 0).
* Test operation
  * `-p,--init_type <0/1/2>` Initialization type: 0=Config (sequential), 1=Parallel, 2=Scalable. Default : 0.
  * `-c,--check <0/1>` Check correctness of results. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make PCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-T,--init_blocking_timeout <seconds>` Timeout for blocking initialization in seconds. Default : 600.
  * `-s,--single_test <0/1>` Run single test mode. Default : 0.
  * `-k,--test_buffer_kind <0/1/2>` Test buffer kind: 0=In-place, 1=Out-of-place, 2=Both. Default : 2.
  * `-R,--local_register <0/1/2>` Enable buffer registration on send/recv buffers: 0=Disabled, 1=Local, 2=Symmetric. Default : 0.
  * `-x,--cta_policy <0/1/2>` Set CTA policy: 0=Default, 1=Efficiency, 2=Zero. Default : -1 (not set).
  * `--init_only <0/1>` Initialization only, skip tests. Default : 0.
  * `--init_blocking <0/1>` Enable blocking initialization. Default : 1.
  * `--buffer_offset <offset bytes>` Send/recv buffer offset in bytes. Default : 0.
  * `--abnormal_mode <0/1/2/3>` Abnormal test mode for testing edge cases. Default : 0.
  * `--target_process <rank>` Target process rank for ACU profiling. Default : -1 (all).
  * `--ncu_profile <0/1>` Enable Asight Compute (NCU) profiling. Default : 0.
  * `--core_dump` Enable core dump generation on failure.
  * `--report_type <0/1/2>` Report detail level: 0=Default, 1=Summary, 2=Detail. Default : 0.
  * `--report_check_filename <filename>` Output file for error reports. Default : none.

## Copyright

PCCL tests is a performance testing tool developed by T-Head (Shanghai) Semiconductor Co., Ltd. and licensed under the BSD license. This project is forked from NVIDIA's NCCL tests. The original NCCL tests remain under the BSD license. All source code and accompanying documentation is copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
