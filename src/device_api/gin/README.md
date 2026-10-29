# GIN P2P benchmarks

These eight benchmarks measure device-side PUT, GET, and signal latency,
bandwidth, and message rate through PCCL GIN.

An MPI build is required. Each invocation uses exactly two MPI ranks and one
PPU per rank. These are performance benchmarks: a successful exit confirms
that the selected operation completed, but it is not a full payload check.

## Build and run

From the repository root, set up the T-Head SAIL SDK environment and point
`PCCL_HOME` to a PCCL installation:

```shell
export PPU_SDK=/path/to/T-Head-SAIL-SDK
source "$PPU_SDK/envsetup.sh"
export PCCL_HOME=/path/to/pccl-install
./build.sh --use_mpi
./build/device_api/gin/throughput/ginPutBW_perf --help
mpirun -np 2 -H host1:1,host2:1 -x PATH -x LD_LIBRARY_PATH \
  -x PCCL_GIN_TYPE=3 \
  ./build/device_api/gin/throughput/ginPutBW_perf \
  -b 64M -e 512M -w 5 -n 30
```

The same executable and matching PCCL, T-Head SAIL SDK, and MPI libraries must
be available to both ranks. Executables are under
`build/device_api/gin/latency/` and `build/device_api/gin/throughput/`.
Without MPI or the required PCCL device interfaces, the normal build omits
these benchmarks and keeps the existing tests.

## Benchmarks

| Executable | Operation |
|---|---|
| `ginGetLatency_ping_perf` | GET followed by local completion |
| `ginPutLatency_ping_perf` | PUT, PUT with remote signal, or PUT with local counter |
| `ginPutLatency_pingPong_perf` | Alternating PUT or PUT with remote signal |
| `ginSignalLatency_ping_perf` | Signal submission and completion |
| `ginSignalLatency_pingPong_perf` | Alternating signal and wait |
| `ginGetBW_perf` | GET bandwidth and message rate |
| `ginPutBW_perf` | PUT, PUT with signal, or PUT with counter bandwidth and message rate |
| `ginSignalBW_perf` | Signal message rate; no payload, so bandwidth is zero |

The PUT-family selection uses `--gin_op put`, `put_signal`, or `put_counter`.
The ping-pong PUT benchmark accepts only `put` and `put_signal`: a local
completion counter cannot tell the remote rank to start its return operation.

## Execution model

`-c` creates that many CTAs and requests at least the same number of GIN
contexts. CTA `blockIdx.x` uses the context with the same index. `-t` selects
the device threads in each CTA; each active thread submits one operation per
iteration. Throughput therefore reports `iterations * CTAs * threads`
messages, doubled by `--gin_bd`. Latency benchmarks always use one CTA and one
thread.

Message buffers are symmetric PCCL windows. For throughput, threads select
slots inside the buffer with `threadIdx.x % (maxbytes / size)`. At the largest
size all threads intentionally reuse the single available slot.

## GIN options

### `--gin_rsm <gpu|cta>`

Selects the GIN resource-sharing mode used when constructing `pcclGin` inside
the kernel. It does not create more contexts or change `-c`:

- `gpu` lets the backend coordinate users of a context at GPU scope.
- `cta` requests CTA-scoped sharing for the threads using that context.

The current PCCL device interface does not expose thread-scoped sharing, so
`--gin_rsm thread` is rejected. GDAKI GET currently uses GPU sharing internally
even when CTA sharing is requested; the configuration line reports
`get_backend_rsm=gpu`. PROXY consumes requests through its GFD path.

### `--gin_tx_depth <depth>`

Sets `pcclDevCommRequirements.ginQueueDepth` when the device communicator is
created. The accepted range is 1 through 32768; the default is 1024.

For GDAKI this is the requested send-queue budget. A payload may be split into
1 GiB chunks, so one logical operation can consume multiple WQEs; PUT_SIGNAL
also needs a signal WQE, while PUT_COUNTER uses a companion counter queue. The
argument checks that one operation fits the requested depth.

For PROXY, the value controls this benchmark's completion-check cadence. It
does not resize the physical QP or GFD ring; `PCCL_GIN_PROXY_QUEUE_SIZE`
controls the GFD ring capacity.

### `--gin_skip_credit_check`

Adds `pcclGinOptFlagsMaySkipCreditCheck` to each throughput operation. It does
not remove completion checking. The kernel estimates WQEs per CTA iteration
and calls `gin.flush()` periodically before half of `--gin_tx_depth` can be
consumed, then flushes again at the end. Unsafe CTA batches are rejected before
MPI starts.

This optimization applies to the GDAKI submission path. PROXY still performs
its per-GFD credit checks.

### `--gin_ag`

Enables `pcclGinOptFlagsAggregateRequests` for throughput. All CTA threads
except the last mark their request as part of an aggregate group; after a CTA
barrier, the last thread submits without the flag and closes the group. This
reduces submission/doorbell work without changing the number of measured
messages.

`-t` must be a multiple of 32. The complete aggregate batch must fit within
half of `--gin_tx_depth`. The current PROXY path accepts the option but retains
per-GFD submission and credit behavior, so it does not provide the GDAKI
aggregation optimization.

### `--local_mem_type` and `--remote_mem_type`

Each option accepts `device` or `host`:

| Operation | Local window | Remote window |
|---|---|---|
| PUT | source | destination |
| GET | destination | source |

`host` uses CPU-backed HOST_NUMA virtual memory that is mapped for both the PPU
and CPU and registered as a symmetric PCCL window. Host memory is supported
only by plain PUT/GET ping and bandwidth benchmarks. Signal, counter, and
ping-pong variants reject host memory rather than silently changing behavior.

### Other operation options

- `--gin_strong_signal` is accepted only by signal and PUT_SIGNAL benchmarks.
  Both selections currently use PCCL's existing strong `pcclGin_SignalInc`, so
  they are not separate weak/strong performance paths.
- `--gin_bd` makes both ranks run throughput kernels and reports the sum of the
  two independently measured rates.

## Sizes, timing, and output

- `-b/-e` select the size range; `K/M/G` use powers of 1024. `-e` is also the
  registered-buffer capacity, limited to 40 GiB.
- `-f` selects multiplicative size steps; non-zero `-i` selects additive steps.
- `-w/-n` select untimed warm-up and measured iterations, defaulting to 50/500.
- Bandwidth is MiB/s and message rate is MPPS.
- Latency is HGGC-event elapsed time divided by iterations and by two. A
  completion-based ping result is therefore a benchmark convention, not a
  direct measurement of physical one-way propagation delay.

## Build scope and limitations

GIN kernels use SIMT branch execution. Category kernels are compiled with a
60-vector-register limit so a 1024-thread CTA remains launchable under the PPU
occupancy guideline; utility objects keep the normal compiler register limit.
Latency and throughput operation objects are archived separately and linked
into the corresponding runners. Existing collective objects and flags are not
changed.

PUT_COUNTER throughput waits for every requested local counter increment before
resetting the counter. Credit-skipping checkpoints also wait for the cumulative
counter value, so counter completion remains inside the measured interval.

The validated backends are selected with `PCCL_GIN_TYPE=3` (GDAKI) and
`PCCL_GIN_TYPE=2` (PROXY). Increasing `-c` uses multiple contexts and peer QPs;
physical network-port distribution remains a property of the selected PCCL
library and topology, not of the benchmark alone.
