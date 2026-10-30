# GIN P2P benchmarks

These eight benchmarks measure device-side PUT, GET, and signal latency,
bandwidth, and message rate. They follow the nccl-tests v2.21.1 benchmark
structure and measurement definitions, adapted to PCCL and HGGC.

An MPI build is required. Each invocation uses exactly two MPI ranks and one
PPU per rank. These are performance benchmarks: a
successful exit does not constitute full payload correctness verification.

## Build and run

From the repository root, with the SAIL SDK environment and `PCCL_HOME` set:

```shell
./build.sh --use_mpi
./build/device_api/gin/throughput/ginPutBW_perf --help
mpirun -np 2 -H host1:1,host2:1 -x LD_LIBRARY_PATH -x PCCL_GIN_TYPE=3 \
  ./build/device_api/gin/throughput/ginPutBW_perf \
  -b 64M -e 512M -w 5 -n 30
```

The same executable and matching PCCL, SDK, and MPI libraries must be available
on both nodes. Executables are under `build/device_api/gin/latency/` and
`build/device_api/gin/throughput/`. Without MPI or sufficiently recent PCCL
headers, the normal build omits these benchmarks and keeps the existing tests.

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
The ping-pong PUT benchmark accepts only `put` and `put_signal`.

## Parameters and metrics

- `-h/--help` displays options without initializing MPI or PCCL.
- `-b/-e` select the size range, defaulting to 4 bytes through 4 MiB. `K/M/G`
  denote powers of 1024. `-e` also sets each registered buffer's capacity, with
  a 40 GiB limit. `-f` selects multiplicative steps; `-i` selects additive steps.
- `-w/-n` select warm-up and measured iterations, defaulting to 50/500.
- `-c/-t` select CTAs and device threads per CTA. Throughput defaults to four
  CTAs and one thread per CTA; an explicit `-c` overrides the default. Latency
  uses one CTA and one thread. Throughput supports up to 1024 threads per CTA.
  These differ from the collective harness's host-thread and correctness flags.
- `--gin_rsm gpu/cta` selects resource sharing. The current PCCL interface does
  not expose the `thread` mode; requests for it receive a diagnostic.
- `--gin_tx_depth` requests a positive send-queue depth, defaulting to 1024.
  `--gin_skip_credit_check` enables periodic completion checks instead of
  per-operation credit checks.
- `--gin_ag` aggregates throughput submissions and requires a thread count
  divisible by 32. `--gin_bd` enables bidirectional throughput.
- `--local_mem_type/--remote_mem_type device/host` select PUT source/destination
  or GET destination/source. HOST is available only for plain PUT/GET ping and
  bandwidth benchmarks, using HOST_NUMA VMM allocation and symmetric windows.
- Bandwidth is in MiB/s and message rate in MPPS. The message count is
  `iterations * CTAs * threads`, doubled for bidirectional runs. Bidirectional
  rates are the sum of the two ranks' independently measured rates.
- Latency is elapsed HGGC-event time divided by iterations and by two, retaining
  the upstream convention. Completion-based ping measurements are not a direct
  measurement of one-way network propagation delay.

## Implementation and scope

GIN objects use SIMT branch execution and are archived per category. Build
stamps track compiler-option changes. The existing collective implementations
and their build settings are unchanged.

Both signal selections use the existing strong `pcclGin_SignalInc` implementation. This does not provide a separate weak-signal
performance path. GDAKI GET uses GPU resource sharing internally even when CTA
sharing is requested; the configuration line identifies this as
`get_backend_rsm=gpu`.

The validated execution path uses GDAKI (`PCCL_GIN_TYPE=3`). These benchmarks
request a positive queue depth and do not override transport initialization
checks. No library-side workaround is included.

Increasing `-c` uses multiple existing GIN contexts and their peer QPs. It does
not guarantee distribution across physical network ports. Network utilization
must be evaluated against the selected device, topology, and transfer direction.
