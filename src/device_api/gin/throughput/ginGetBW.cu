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
__global__ void ginGetBwKernel(pcclDevComm comm, pcclWindow_t rWindow, pcclWindow_t lWindow, size_t numElems,
                               int iters, int queueDepth, size_t maxElems, bool isRemoteBufDev, bool isLocalBufDev) {
  const int tag = blockIdx.x;
  pcclTeam team = pcclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  pcclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;
  const size_t myBytes = numElems * sizeof(int);

  pcclSymPtr<int> rbuf = pcclSymPtr<int>(rWindow, offset * sizeof(int));
  pcclSymPtr<int> lbuf = pcclSymPtr<int>(lWindow, offset * sizeof(int));

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;
  const int lastActive = (int)(blockDim.x - 1);
  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        ginGetDispatch(isLocalBufDev, isRemoteBufDev, gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset,
                       myBytes, pcclCoopThread{}, pcclGin_None{}, optFlags | pcclGinOptFlagsAggregateRequests);

      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        ginGetDispatch(isLocalBufDev, isRemoteBufDev, gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset,
                       myBytes, pcclCoopThread{}, pcclGin_None{}, optFlags);

      }
    } else {
      ginGetDispatch(isLocalBufDev, isRemoteBufDev, gin, team, peer, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset,
                     myBytes, pcclCoopThread{}, pcclGin_None{}, optFlags);

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
static void ginGetBWLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                               const ginArgs_t* args, size_t numElems, int iters) {
  pcclWindow_t rWindow = args->remoteMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  pcclWindow_t lWindow = args->localMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
  const bool isRemoteBufDev = args->remoteMemoryType == pcclGinMemoryDevice;
  const bool isLocalBufDev = args->localMemoryType == pcclGinMemoryDevice;
#define LAUNCH_BW_GET(SKIP, AG) \
  ginGetBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>( \
    ctx->dcomm, rWindow, lWindow, numElems, iters, queueDepth, maxElems, isRemoteBufDev, isLocalBufDev)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_GET(true, true);
    else                              LAUNCH_BW_GET(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_GET(false, true);
    else                              LAUNCH_BW_GET(false, false);
  }

#undef LAUNCH_BW_GET
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinGetBWLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginGetBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginGetBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
