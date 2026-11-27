/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "common.h"
#include "gin_context.h"
#include <sched.h>

/*
 * A binary can cover several operations (the put benchmarks select between put,
 * put+signal and put+counter via --gin_op), so name the resolved configuration
 * before the table.
 */
static void printConfiguration(const ginArgs_t* args, const ginBenchmark_t* bench, const ginContext_t* ctx) {
  const auto type = ctx->dcomm.ginNetDeviceTypes[0];
  const char* backend;
  switch (type) {
    case PCCL_NET_DEVICE_GIN_GDAKI: backend = "GDAKI"; break;
    case PCCL_NET_DEVICE_GIN_PROXY: backend = "PROXY"; break;
#if defined(PCCLX_GIN_TROCE_ENABLE) && PCCLX_GIN_TROCE_ENABLE
    case PCCLX_NET_DEVICE_GIN_TROCE: backend = "TROCE"; break;
#endif
    default: backend = "unknown"; break;
  }
  printf("# %s: backend=%s, rsm=%s", bench->name ? bench->name : "gin benchmark",
         backend, pcclTestGinRsmName(args->ginRsm));
  if (args->signalType != pcclTestSignalUnset)
    printf(", signal_type=%s", args->signalType == pcclTestSignalSet ? "set" : "inc");
  if (type == PCCL_NET_DEVICE_GIN_GDAKI && args->ginRsm == pcclGinRsmCta && bench->name && strstr(bench->name, "GIN GET"))
    printf(", get_backend_rsm=gpu");
  if (args->ginStrongSignal) printf(", strong_signal");
  if (args->ginSkipCreditCheck) printf(", skip_credit_check");
  if (args->ginAggregateRequests) printf(", aggregate_requests");
  if (args->ginBidirectional) printf(", bidirectional");
  printf(", tx_depth=%d, contexts=%u, connections=%u", args->queueDepth,
         (unsigned)ctx->dcomm.ginContextCount, (unsigned)ctx->dcomm.ginConnectionCount);
  printf(", local mem type=%s, remote mem type=%s", pcclTestGinMemoryTypeName(args->localMemoryType), pcclTestGinMemoryTypeName(args->remoteMemoryType));
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) printf(", ctas=%d, threads=%d", args->numCtas, args->numThreads);
  printf("\n");
}

static void printHeader(const ginBenchmark_t* bench) {
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) {
    printf("%12s  %14s  %18s  %20s\n", "Size(B)", "Num_Messages", "Bandwidth(MiB/s)",
           "Message_Rate(MPPS)");
  } else {
    printf("%12s  %8s  %14s\n", "Size(B)", "Iters", "Latency(us)");
  }
}

static void printThroughputInformation(size_t size, double numMessages, double bandwidthMiBps,
                                       double messageRateMpps) {
  printf("%12zu  %14.0f  %18.3f  %20.3f\n", size, numMessages, bandwidthMiBps, messageRateMpps);
}

static void printLatencyInformation(size_t size, int iters, double latencyUs) {
  printf("%12zu  %8d  %14.3f\n", size, iters, latencyUs);
}

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  memset(hostname, 0, maxlen);
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen && hostname[i] != '\0'; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

static size_t nextSize(const ginArgs_t* args, size_t size) {
  if (args->stepBytes != 0)
    return args->stepBytes > SIZE_MAX - size ? 0 : size + args->stepBytes;
  size_t factor = args->stepFactor;
  if (factor <= 1) factor = 2;
  return size > SIZE_MAX / factor ? 0 : size * factor;
}

static double throughputBytesPerMessage(const ginBenchmark_t* bench, size_t size) {
  switch (bench->payloadMode) {
  case GIN_THROUGHPUT_PAYLOAD_NONE:
    return 0.0;
  case GIN_THROUGHPUT_PAYLOAD_FIXED:
    return (double)bench->payloadFixedBytes;
  case GIN_THROUGHPUT_PAYLOAD_SIZE:
  default:
    return (double)size;
  }
}

static constexpr double signalTimeoutSeconds = 10.0;

#if PCCL_TEST_GIN_HAS_SIGNAL_SET
static void checkProxySet(pcclComm_t comm, const ginContext_t* ctx, int contexts, hggcStream_t stream);
#endif
static void prepareSignals(const ginContext_t* ctx, int contexts, int slots, hggcStream_t stream);
static void waitForSignalSlots(const ginContext_t* ctx, hggcStream_t stream, int contexts, int slots,
                               uint64_t expected, int* mismatch, const char* phase);
static void completeSignalPhase(const ginContext_t* ctx, hggcStream_t stream, int slots, int* mismatch,
                                const ginArgs_t* args, int iters, int rank, size_t size, bool measured);

void pcclTestGinPerfRun(int argc, char** argv, const ginArgs_t* args, const ginBenchmark_t* bench) {

  setlinebuf(stdout);
  int rank = 0, nRanks = 0;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK_FATAL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK_FATAL(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2) {
    if (rank == 0) fprintf(stderr, "Error: GIN perf benchmarks require exactly 2 MPI ranks (got %d)\n", nRanks);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK_FATAL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r = 0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  HGGCCHECK_FATAL(hggcSetDevice(dev));

  pcclUniqueId id;
  if (rank == 0) PCCLCHECK_FATAL(pcclGetUniqueId(&id));
  MPICHECK_FATAL(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  hggcStream_t stream;
  HGGCCHECK_FATAL(hggcStreamCreate(&stream));

  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  pcclComm_t comm;
  PCCLCHECK_FATAL(pcclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ginContext_t ctx = {};
  pcclTestGinDevCommCreate(comm, args, &ctx);
  // No rank may start a polling kernel until every symmetric window has a
  // deterministic value and is registered on both peers.
  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

  const bool bidir = args->ginBidirectional;
  const bool participates = (bench->type == GIN_BENCHMARK_TYPE_PING_PONG || bidir) || rank == 0;
  const int timingRank = bench->type == GIN_BENCHMARK_TYPE_PING_PONG ? 1 : 0;
  const bool measures = (rank == timingRank) || bidir;
  const bool signalsInUse = args->signalType != pcclTestSignalUnset;
  const int slots = args->signalType == pcclTestSignalSet ? args->numThreads : 1;
  const bool checkReceiver = signalsInUse && (rank == 1 || bidir || bench->type == GIN_BENCHMARK_TYPE_PING_PONG);
  int* mismatch = nullptr;
  if (signalsInUse) HGGCCHECK_FATAL(hggcMalloc((void**)&mismatch, sizeof(int)));
#if PCCL_TEST_GIN_HAS_SIGNAL_SET
  if (args->signalType == pcclTestSignalSet && ctx.dcomm.ginNetDeviceTypes[0] == PCCL_NET_DEVICE_GIN_PROXY) {
    prepareSignals(&ctx, args->numCtas, 1, stream);
    checkProxySet(comm, &ctx, args->numCtas, stream);
    waitForSignalSlots(&ctx, stream, args->numCtas, 1, 1, mismatch, "PROXY SET probe");
    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
  }
#endif

  hggcEvent_t start, stop;
  if (measures) {
    HGGCCHECK_FATAL(hggcEventCreate(&start));
    HGGCCHECK_FATAL(hggcEventCreate(&stop));
  }

  if (rank == 0) {
    printConfiguration(args, bench, &ctx);
    printHeader(bench);
  }

  for (size_t size = args->minBytes; size <= args->maxBytes;) {
    size_t numElems = size / sizeof(int);

    //-----Warmup-----
    if (signalsInUse && args->warmupIters > 0) prepareSignals(&ctx, args->numCtas, slots, stream);
    if (args->warmupIters > 0 && participates) {
      bench->run(&ctx, stream, args, numElems, args->warmupIters);
      HGGCCHECK_FATAL(hggcStreamSynchronize(stream));
    }

    // Sender-side flush may complete before the signal is visible on its receiver.
    if (signalsInUse && args->warmupIters > 0)
      completeSignalPhase(&ctx, stream, slots, checkReceiver ? mismatch : nullptr,
                          args, args->warmupIters, rank, size, false);
    else if (!signalsInUse)
      MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    //-----Benchmark-----
    if (signalsInUse) prepareSignals(&ctx, args->numCtas, slots, stream);

    if (measures) HGGCCHECK_FATAL(hggcEventRecord(start, stream));
    if (participates) bench->run(&ctx, stream, args, numElems, args->iters);
    if (measures) HGGCCHECK_FATAL(hggcEventRecord(stop, stream));
    if (participates) HGGCCHECK_FATAL(hggcStreamSynchronize(stream));

    if (signalsInUse)
      completeSignalPhase(&ctx, stream, slots, checkReceiver ? mismatch : nullptr,
                          args, args->iters, rank, size, true);

    if (bench->type == GIN_BENCHMARK_TYPE_PING || bench->type == GIN_BENCHMARK_TYPE_PING_PONG) {
      // Latency: rank 0 reports (bidirectional is not supported for latency).
      float milliseconds = 0.0f;
      if (rank == timingRank) HGGCCHECK_FATAL(hggcEventElapsedTime(&milliseconds, start, stop));
      if (bench->type == GIN_BENCHMARK_TYPE_PING_PONG)
        MPICHECK_FATAL(MPI_Bcast(&milliseconds, 1, MPI_FLOAT, timingRank, MPI_COMM_WORLD));
      if (rank == 0) {
        double rttUs = (double)milliseconds * 1000.0 / (double)args->iters;
        double latencyUs = rttUs / 2.0;
        printLatencyInformation(size, args->iters, latencyUs);
      }
    } else {
      double numMessages = 0.0, bandwidthMiBps = 0.0, messageRateMpps = 0.0;
      if (measures) {
        float milliseconds = 0.0f;
        HGGCCHECK_FATAL(hggcEventElapsedTime(&milliseconds, start, stop));
        double seconds = (double)milliseconds / 1e3;
        numMessages = (double)args->iters * (double)args->numCtas * (double)args->numThreads;
        double bytes = numMessages * throughputBytesPerMessage(bench, size);
        bandwidthMiBps = bytes / (double)(1ULL << 20) / seconds;
        messageRateMpps = numMessages / 1e6 / seconds;
      }

      if (bidir) {
        double localMetrics[3] = {numMessages, bandwidthMiBps, messageRateMpps};
        double summedMetrics[3] = {0.0, 0.0, 0.0};
        MPICHECK_FATAL(MPI_Reduce(localMetrics, summedMetrics, 3, MPI_DOUBLE, MPI_SUM, 0,
                                  MPI_COMM_WORLD));
        numMessages = summedMetrics[0];
        bandwidthMiBps = summedMetrics[1];
        messageRateMpps = summedMetrics[2];
      }

      if (rank == 0) printThroughputInformation(size, numMessages, bandwidthMiBps, messageRateMpps);
    }

    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    size_t next = nextSize(args, size);
    if (next <= size) break;
    size = next;
  }

  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
  pcclTestGinDevCommDestroy(comm, &ctx);
  if (mismatch) HGGCCHECK_FATAL(hggcFree(mismatch));
  PCCLCHECK_FATAL(pcclCommDestroy(comm));
  if (measures) {
    HGGCCHECK_FATAL(hggcEventDestroy(start));
    HGGCCHECK_FATAL(hggcEventDestroy(stop));
  }
  HGGCCHECK_FATAL(hggcStreamDestroy(stream));
  MPICHECK_FATAL(MPI_Finalize());
}

#if PCCL_TEST_GIN_HAS_SIGNAL_SET
static __global__ void probeProxySet(pcclDevComm comm) {
  pcclGin gin(comm, blockIdx.x);
  gin.signal(pcclTeamWorld(comm), comm.rank ^ 1, pcclxGin_SignalSet{0, 1});
  gin.flush(pcclCoopThread{});
}

// PROXY does not advertise the underlying provider's signal operations. Surface
// its asynchronous rejection before entering a timed kernel or a polling wait.
static void checkProxySet(pcclComm_t comm, const ginContext_t* ctx, int contexts, hggcStream_t stream) {
  probeProxySet<<<contexts, 1, 0, stream>>>(ctx->dcomm);
  HGGCCHECK_FATAL(hggcGetLastError());
  const double deadline = MPI_Wtime() + signalTimeoutSeconds;
  hggcError_t status;
  do {
    status = hggcStreamQuery(stream);
    pcclResult_t result;
    PCCLCHECK_FATAL(pcclCommGetAsyncError(comm, &result));
    if (result != pcclSuccess) {
      fprintf(stderr, "Error: the GIN PROXY provider failed the SET probe: %s\n", pcclGetErrorString(result));
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (status == hggcErrorNotReady) {
      if (MPI_Wtime() >= deadline) {
        fprintf(stderr, "Error: timed out submitting the GIN PROXY SET probe\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      sched_yield();
    }
  } while (status == hggcErrorNotReady);
  HGGCCHECK_FATAL(status);
  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
}
#endif

// Mirror the measured launch: blockIdx.x selects the context, threadIdx.x the signal.
static __global__ void resetSignalSlots(pcclDevComm comm) {
  pcclGin gin(comm, blockIdx.x);
  gin.resetSignal(threadIdx.x);
}

static __global__ void verifySignalSlots(pcclDevComm comm, uint64_t expected, int* mismatch, bool diagnose) {
  pcclGin gin(comm, blockIdx.x);
  uint64_t value = gin.readSignal(threadIdx.x);
  if (value != expected && atomicCAS(mismatch, 0, 1) == 0 && diagnose) {
    printf("Signal mismatch: rank=%d context=%u slot=%u expected=%llu observed=%llu\n",
           comm.rank, blockIdx.x, threadIdx.x, (unsigned long long)expected, (unsigned long long)value);
  }
}

static void prepareSignals(const ginContext_t* ctx, int contexts, int slots, hggcStream_t stream) {
  resetSignalSlots<<<contexts, slots, 0, stream>>>(ctx->dcomm);
  HGGCCHECK_FATAL(hggcGetLastError());
  HGGCCHECK_FATAL(hggcStreamSynchronize(stream));
  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
}

static void waitForSignalSlots(const ginContext_t* ctx, hggcStream_t stream, int contexts, int slots,
                               uint64_t expected, int* mismatch, const char* phase) {
  const double deadline = MPI_Wtime() + signalTimeoutSeconds;
  for (;;) {
    const bool expired = MPI_Wtime() >= deadline;
    HGGCCHECK_FATAL(hggcMemsetAsync(mismatch, 0, sizeof(int), stream));
    verifySignalSlots<<<contexts, slots, 0, stream>>>(ctx->dcomm, expected, mismatch, expired);
    HGGCCHECK_FATAL(hggcGetLastError());
    HGGCCHECK_FATAL(hggcStreamSynchronize(stream));
    int failed;
    HGGCCHECK_FATAL(hggcMemcpy(&failed, mismatch, sizeof(failed), hggcMemcpyDeviceToHost));
    if (!failed) return;
    if (expired) {
      fprintf(stderr, "Error: timed out waiting for %s signals on rank %d\n", phase, ctx->dcomm.rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    sched_yield();
  }
}

static void completeSignalPhase(const ginContext_t* ctx, hggcStream_t stream, int slots, int* mismatch,
                                const ginArgs_t* args, int iters, int rank, size_t size, bool measured) {
  // Start the visibility deadline only after all senders finish this phase.
  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
  if (mismatch) {
    const bool setSignal = args->signalType == pcclTestSignalSet;
    const uint64_t expected = setSignal ? ginSetValue(iters - 1) : (uint64_t)iters * args->numThreads;
    waitForSignalSlots(ctx, stream, args->numCtas, slots, expected, mismatch, measured ? "measured" : "warm-up");
    if (measured)
      printf("# %s verified: rank=%d bytes=%zu contexts=%d slots_per_context=%d value=0x%llx\n",
             setSignal ? "SET" : "INC", rank, size, args->numCtas, slots, (unsigned long long)expected);
  }
  // All receivers must finish before a rank resets a slot or reports a measurement.
  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
}
