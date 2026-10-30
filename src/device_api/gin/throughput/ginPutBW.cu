/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "pccl.h"
#include "pccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, pcclGinResourceSharingMode rsm>
__global__ void ginPutBwKernel(pcclDevComm comm, pcclWindow_t dWindow, pcclWindow_t sWindow, size_t numElems,
                               int iters, int queueDepth, size_t maxElems, bool isDestBufDev, bool isSourceBufDev) {
  const int tag = blockIdx.x;
  pcclTeam team = pcclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  pcclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;

  pcclSymPtr<int> dbuf = pcclSymPtr<int>(dWindow, offset * sizeof(int));
  pcclSymPtr<int> sbuf = pcclSymPtr<int>(sWindow, offset * sizeof(int));

  const int lastActive = (int)(blockDim.x - 1);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        ginPutDispatch(isSourceBufDev, isDestBufDev, gin, team, peer, dbuf, sbuf, numElems,
                       pcclGin_None{}, pcclGin_None{}, pcclCoopThread{}, pcclGin_None{},
                       hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags | pcclGinOptFlagsAggregateRequests);

      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        ginPutDispatch(isSourceBufDev, isDestBufDev, gin, team, peer, dbuf, sbuf, numElems,
                       pcclGin_None{}, pcclGin_None{}, pcclCoopThread{}, pcclGin_None{},
                       hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);

      }
    } else {
      ginPutDispatch(isSourceBufDev, isDestBufDev, gin, team, peer, dbuf, sbuf, numElems,
                     pcclGin_None{}, pcclGin_None{}, pcclCoopThread{}, pcclGin_None{},
                     hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);

    }
    if constexpr (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(pcclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(pcclCoopCta{});
}

template <pcclGinResourceSharingMode rsm>
static void ginPutBWLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                               const ginArgs_t* args, size_t numElems, int iters) {
  pcclWindow_t dWindow = args->remoteMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  pcclWindow_t sWindow = args->localMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
  bool isDestBufDev = args->remoteMemoryType == pcclGinMemoryDevice;
  bool isSourceBufDev = args->localMemoryType == pcclGinMemoryDevice;
#define LAUNCH_BW_PUT(SKIP, AG) \
  ginPutBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>( \
    ctx->dcomm, dWindow, sWindow, numElems, iters, queueDepth, maxElems, isDestBufDev, isSourceBufDev)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT(true, true);
    else                              LAUNCH_BW_PUT(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT(false, true);
    else                              LAUNCH_BW_PUT(false, false);
  }

#undef LAUNCH_BW_PUT
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutBWLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
