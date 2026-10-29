/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <type_traits>
#include <unistd.h>

#include "hggc_runtime.h"
#include "pccl.h"
#include "pccl_tests_version.h"
#include "pccl_device.h"
#include "mpi.h"
#include "args.h"
#include "gin_context.h"

#if PCCL_CHECK_HGGCCC

PCCL_DEVICE_INLINE void ginGetDevice(pcclGin& gin, pcclTeam team, int peer, pcclWindow_t remoteWnd,
                                     size_t remoteOffset, pcclWindow_t localWnd, size_t localOffset, size_t bytes,
                                     pcclCoopThread coop, pcclGin_None descriptor, uint32_t optFlags) {
  gin.get(team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags,
          pcclGin_SegmentDevice{});
}
PCCL_DEVICE_INLINE void ginGetHostNuma(pcclGin& gin, pcclTeam team, int peer, pcclWindow_t remoteWnd,
                                       size_t remoteOffset, pcclWindow_t localWnd, size_t localOffset, size_t bytes,
                                       pcclCoopThread coop, pcclGin_None descriptor, uint32_t optFlags) {
  gin.get(team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags,
          pcclGin_SegmentHostNuma{});
}
PCCL_DEVICE_INLINE void ginGetMixed(pcclGin& gin, pcclTeam team, int peer, pcclWindow_t remoteWnd,
                                    size_t remoteOffset, pcclWindow_t localWnd, size_t localOffset, size_t bytes,
                                    pcclCoopThread coop, pcclGin_None descriptor, uint32_t optFlags) {
  gin.get(team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags,
          pcclGin_SegmentMixed{});
}


PCCL_DEVICE_INLINE void ginGetDispatch(bool isLocalBufDev, bool isRemoteBufDev,
                                       pcclGin& gin, pcclTeam team, int peer, pcclWindow_t remoteWnd,
                                       size_t remoteOffset, pcclWindow_t localWnd, size_t localOffset,
                                       size_t bytes, pcclCoopThread coop, pcclGin_None descriptor, uint32_t optFlags) {
  if (isLocalBufDev && isRemoteBufDev)
    ginGetDevice(gin, team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags);
  else if (!isLocalBufDev && !isRemoteBufDev)
    ginGetHostNuma(gin, team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags);
  else
    ginGetMixed(gin, team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coop, descriptor, optFlags);
}

PCCL_DEVICE_INLINE void ginPutDevice(pcclGin& gin, pcclTeam team, int peer, pcclSymPtr<int> dstElts,
                                     pcclSymPtr<int> srcElts, size_t nElts, pcclGin_None remoteAction,
                                     pcclGin_None localAction, pcclCoopThread coop, pcclGin_None descriptor,
                                     hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease, uint32_t optFlags) {
  gin.put(team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
          givenRelease, requiredRelease, optFlags, pcclGin_SegmentDevice{});
}
PCCL_DEVICE_INLINE void ginPutHostNuma(pcclGin& gin, pcclTeam team, int peer, pcclSymPtr<int> dstElts,
                                       pcclSymPtr<int> srcElts, size_t nElts, pcclGin_None remoteAction,
                                       pcclGin_None localAction, pcclCoopThread coop, pcclGin_None descriptor,
                                       hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease, uint32_t optFlags) {
  gin.put(team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
          givenRelease, requiredRelease, optFlags, pcclGin_SegmentHostNuma{});
}
PCCL_DEVICE_INLINE void ginPutMixed(pcclGin& gin, pcclTeam team, int peer, pcclSymPtr<int> dstElts,
                                    pcclSymPtr<int> srcElts, size_t nElts, pcclGin_None remoteAction,
                                    pcclGin_None localAction, pcclCoopThread coop, pcclGin_None descriptor,
                                    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease, uint32_t optFlags) {
  gin.put(team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
          givenRelease, requiredRelease, optFlags, pcclGin_SegmentMixed{});
}


PCCL_DEVICE_INLINE void ginPutDispatch(bool isSourceBufDev, bool isDestBufDev,
                                       pcclGin& gin, pcclTeam team, int peer, pcclSymPtr<int> dstElts,
                                       pcclSymPtr<int> srcElts, size_t nElts, pcclGin_None remoteAction,
                                       pcclGin_None localAction, pcclCoopThread coop, pcclGin_None descriptor,
                                       hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease, uint32_t optFlags) {
  if (isSourceBufDev && isDestBufDev)
    ginPutDevice(gin, team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
                 givenRelease, requiredRelease, optFlags);
  else if (!isSourceBufDev && !isDestBufDev)
    ginPutHostNuma(gin, team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
                   givenRelease, requiredRelease, optFlags);
  else
    ginPutMixed(gin, team, peer, dstElts, srcElts, nElts, remoteAction, localAction, coop, descriptor,
                givenRelease, requiredRelease, optFlags);
}

#endif // PCCL_CHECK_HGGCCC

/* Pre-MPI: use plain exit so the caller sees a clean error before MPI is up. */
#define MPICHECK(cmd) do { \
  int _e = (cmd); \
  if (_e != MPI_SUCCESS) { \
    fprintf(stderr, "Failed: MPI error %s:%d '%d'\n", \
            __FILE__, __LINE__, _e); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define HGGCCHECK(cmd) do { \
  hggcError_t _e = (cmd); \
  if (_e != hggcSuccess) { \
    fprintf(stderr, "Failed: HGGC error %s:%d '%s'\n", \
            __FILE__, __LINE__, hggcGetErrorString(_e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define PCCLCHECK(cmd) do { \
  pcclResult_t _r = (cmd); \
  if (_r != pcclSuccess) { \
    fprintf(stderr, "Failed: PCCL error %s:%d '%s'\n", \
            __FILE__, __LINE__, pcclGetErrorString(_r)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

/* Post-MPI: use MPI_Abort so all ranks are torn down consistently. */
#define MPICHECK_FATAL(cmd) do { \
  int _e = (cmd); \
  if (_e != MPI_SUCCESS) { \
    fprintf(stderr, "Fatal: MPI error %s:%d '%d'\n", \
            __FILE__, __LINE__, _e); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define HGGCCHECK_FATAL(cmd) do { \
  hggcError_t _e = (cmd); \
  if (_e != hggcSuccess) { \
    fprintf(stderr, "Fatal: HGGC error %s:%d '%s'\n", \
            __FILE__, __LINE__, hggcGetErrorString(_e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define PCCLCHECK_FATAL(cmd) do { \
  pcclResult_t _r = (cmd); \
  if (_r != pcclSuccess) { \
    fprintf(stderr, "Fatal: PCCL error %s:%d '%s'\n", \
            __FILE__, __LINE__, pcclGetErrorString(_r)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

/*
 * Benchmark callback invoked by pcclTestGinPerfRun for every (size, iters) point.
 *
 *   ctx – Device comm context
 *   stream – HGGC stream to launch into
 *   args – parsed CLI args
 *   numElems – elements (ints) to transfer this size point
 *   iters – number of iterations to perform this call
 */
typedef void (*ginRunFn_t)(
    const ginContext_t* ctx,
    hggcStream_t stream,
    const ginArgs_t* args,
    size_t numElems,
    int iters
);

/*
 * Called on rank 0 only, once per size point, with the elapsed time of the
 * measured (non-warmup) phase. Print whatever metric this category cares
 * about (bandwidth divides bytes by time, latency divides time by iters, ...).
 */
typedef void (*ginReportFn_t)(size_t size, int iters, double milliseconds);

enum ginBenchmarkType_t {
  GIN_BENCHMARK_TYPE_PING,
  GIN_BENCHMARK_TYPE_PING_PONG,
  GIN_BENCHMARK_TYPE_THROUGHPUT,
};


enum ginThroughputPayload_t {
  GIN_THROUGHPUT_PAYLOAD_SIZE,
  GIN_THROUGHPUT_PAYLOAD_NONE,
  GIN_THROUGHPUT_PAYLOAD_FIXED,
};

/*
 * Bundles everything that differs between benchmark categories/APIs so a
 * single pcclTestGinPerfRun() can serve all of them:
 *
 *   run     – the kernel-launch callback (see ginRunFn_t above).
 *   name    – human-readable test name, reported above the results table.
 *   type    – benchmark category; ping-pong launches on both ranks, ping and
 *             throughput launch on rank 0 only.
 *   payloadMode – throughput only: how payload bytes per message are counted.
 *   payloadFixedBytes – when payloadMode is FIXED, bytes per message (e.g. sizeof(T)).
 */
typedef struct {
  ginRunFn_t run;
  const char* name;
  ginBenchmarkType_t type;
  ginThroughputPayload_t payloadMode;
  size_t payloadFixedBytes;
} ginBenchmark_t;

/*
 * Full perf harness: setup, warmup+measured size sweep, teardown. Shared by
 * every benchmark category (bandwidth, message rate, ping latency, ping-pong
 * latency) and every GIN API (Put, Get, Signal, PutSignal, ...) via the
 * ginBenchmark_t callbacks/flags above. pcclTestGinParseArgs must have been
 * called before this function.
 */
void pcclTestGinPerfRun(int argc, char** argv, const ginArgs_t* args, const ginBenchmark_t* bench);
