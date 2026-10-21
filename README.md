# PCCL Tests

PCCL Tests provides performance and correctness benchmarks for [PCCL](https://github.com/t-head/pccl) collective
operations on Zhenwu PPU devices.

## Build

Set up the T-Head SAIL SDK environment and set `PCCL_HOME` to the PCCL
installation directory before building:

```shell
export PPU_SDK=/usr/local/PPU_SDK  # or your SDK path
source "$PPU_SDK/envsetup.sh"
export PCCL_HOME=/path/to/pccl-install
./build.sh
```

Multi-process tests require Open MPI. To build with MPI support:

```shell
./build.sh --use_mpi
```

Use `--rebuild` when changing build options to recompile existing objects.

Benchmark executables are generated in `build/`.

For build options, run:

```shell
./build.sh --help
```

## Usage

Use MPI to launch multiple processes. Set the number of host threads per
process with `-t` and the number of PPU devices per thread with `-g`.

### Quick Examples

Run AllReduce on eight PPU devices and sweep message sizes from 8 bytes to
128 MiB, doubling the size after each measurement:

```shell
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

The benchmarks must be built with `--use_mpi` before they can be launched with
MPI. The following example runs 64 ranks across eight nodes, using one PPU
device per process:

```shell
mpirun -np 64 -N 8 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

For reproducible performance results, configure MPI process binding and
CPU/device affinity explicitly.

### Performance

The default report includes operation time in microseconds, algorithm
bandwidth (`algbw`), bus bandwidth (`busbw`), and the number of incorrect
elements (`#wrong`) for in-place and out-of-place tests.

- `algbw` is the message size divided by operation time.
- `busbw` applies a collective-specific correction factor so results can be
  compared more directly with the communication bandwidth of the system.
- `#wrong` must be zero for a successful correctness run.

See [Performance](doc/PERFORMANCE.md) for the formulas and timing-mode
details.

### Arguments

All benchmark executables share the same common options. Run any benchmark
with `-h` or `--help` to see the options available in your build.

- Number of devices

  - `-t, --nthreads <N>` Host threads per process. Default: `1`.
  - `-g, --ngpus <N>` PPU devices per thread. Default: `1`.

- Message sizes

  - `-b, --minbytes <size>` Minimum message size. Default: `32M`.
  - `-e, --maxbytes <size>` Maximum message size. Default: `32M`.
  - `-i, --stepbytes <bytes>` Fixed increment between message sizes, in bytes.
    Default: `1048576` (1 MiB).
  - `-f, --stepfactor <factor>` Multiplicative increment; values greater than
    `1` override `--stepbytes`. Default: `1` (disabled).
  - `--sizelist <sizes>` Comma-separated message sizes. Default: unset.
  - `--random_size_num <N>` Generate `N` random message sizes in the configured
    range. Default: `0`.

  `--minbytes`, `--maxbytes`, and `--sizelist` accept `K`, `M`, and `G`
  suffixes using powers of 1024. `--stepbytes` takes an integer byte count.

- PCCL operations

  - `-o, --op <op>` Reduction operation: `sum`, `prod`, `min`, `max`, `avg`,
    `mulsum`, or `all`. Default: `sum`.
  - `-d, --datatype <type>` Data type or `all`. Default: `float`.
    The base types are `int8`, `uint8`, `int32`, `uint32`, `int64`, `uint64`,
    `half`, `float`, and `double`. Availability of `bf16`, `f8e4m3`, and
    `f8e5m2` depends on the PCCL and HGGC versions.
  - `-r, --root <rank>` Root rank for rooted collectives. Default: `0`.

- Performance and reporting

  - `-n, --iters <N>` Timed iterations. Default: `20`.
  - `-w, --warmup_iters <N>` Warm-up iterations. Default: `5`.
  - `-m, --agg_iters <N>` Operations aggregated in each iteration. Default: `1`.
  - `-a, --average <0/1/2/3>` Report rank 0, average, minimum, or maximum time.
    Default: `2`.
  - `--elapsed_type <0/1/2>` CPU, HGGC event, or HGPTI timing. Default: `0`.
  - `--api_time` Report host API overhead in the default report mode
    (`--report_type 0`). Default: disabled.
  - `--acu_profile <0/1>` Enable Asight Compute profiling. Default: `0`.
  - `--target_process <rank>` Process rank to profile with `--acu_profile`;
    `-1` selects all processes. Default: `-1`.
  - `--report_type <0/1/2>` Default, summary, or detailed report. Default: `0`.
  - `--report_check_filename <prefix>` Write correctness failures to
    `<prefix>_<rank>.log`. Default: unset.

- Test operation

  - `-c, --check <0/1>` Validate results. Default: `1`.
  - `-k, --test_buffer_kind <0/1/2>` In-place only, out-of-place only, or both.
    Default: `2`.
  - `-z, --blocking <0/1>` Wait and synchronize after each collective.
    Default: `0`.
  - `-G, --hggcgraph <N>` Capture iterations in an HGGC graph and replay it
    `N` times. Default: `0` (disabled).
  - `-R, --local_register <0/1/2>` Disable buffer registration, use local
    registration, or use symmetric registration. Default: `0`.
  - `-x, --cta_policy <0/1/2>` CTA policy: default, efficiency, or zero.
    Default: unset.
  - `-D, --device_implementation <N>` Device-side implementation ID; `0` uses
    the standard collective API. Default: `0`. Values greater than `0`
    require PCCL 2.2 or later and symmetric registration (`-R 2`).
    Support varies by benchmark.
  - `-H, --host_rma_implementation <N>` Use host one-sided RMA
    (`pcclPutSignal`/`pcclWaitSignal`) instead of the built-in collective.
    Supported by sendrecv, alltoall, all_gather, broadcast, gather, and
    scatter. `1` uses a single context; values greater than `1` distribute
    RMA traffic across contexts. Requires PCCL 2.2 or later, symmetric
    registration (`-R 2`), and is mutually exclusive with `-D`.
  - `-V, --device_cta_count <N>` Number of CTAs for device-side implementations,
    from `1` to `128`. Default: `16`.

- Initialization

  - `-p, --init_type <0/1/2>` Configured, parallel, or scalable communicator
    initialization. Default: `0`.
  - `--init_blocking <0/1>` Use nonblocking or blocking communicator
    initialization. Default: `1`.
  - `-T, --init_blocking_timeout <seconds>` Timeout when polling a nonblocking
    communicator; `0` disables the timeout. Default: `600`.
  - `--init_only <0/1>` Initialize communicators without running benchmarks.
    Default: `0`.

  `--init_blocking` and `--cta_policy` configure multi-process initialization
  with `--init_type 0` or `2`. For a single process, these modes use
  `pcclCommInitAll` instead.

- Debugging and failure testing

  - `-s, --single_test <0/1>` Skip the extra untimed collective before each
    measurement; warm-up iterations still run. Default: `0`.
  - `--buffer_offset <bytes>` Offset send/receive buffers on even-numbered
    devices. Rounded down to a multiple of the data type size. Default: `0`.
  - `--core_dump <0/1>` Request unlimited core dump size from the operating
    system. Default: `0`.
  - `--abnormal_mode <0/1/2/3>` Failure-injection mode: disabled, datatype
    mismatch, count mismatch, or delayed processes. Default: `0`.
    These modes can deliberately hang or delay a run.

### Running multiple operations in parallel

PCCL Tests can divide PPU devices into groups that run the same operation
in parallel. With an MPI build (`--use_mpi`), set `NCCL_TESTS_SPLIT_MASK` to
a hexadecimal mask. Processes with the same `MPI rank & mask` value form
a group, shown next to each device at startup.

For example, on multiple nodes with eight PPU devices per node, one device
per MPI process (`-t 1 -g 1`), and consecutive MPI ranks on each node:

- `NCCL_TESTS_SPLIT_MASK=0x7`: Run eight groups in parallel, each with one
  device per node (inter-node communication).
- `NCCL_TESTS_SPLIT_MASK=0x1`: Run two groups containing even and odd MPI
  ranks, respectively.

## Copyright

Licensed under the BSD 3-Clause license. See [LICENSE.txt](LICENSE.txt).

Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.

Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All Rights Reserved.
