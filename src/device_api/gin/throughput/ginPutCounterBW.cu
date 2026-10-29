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
__global__ void ginPutCounterBwKernel(pcclDevComm comm, pcclWindow_t devWindow, int iters, size_t numElems,
                                        int queueDepth, size_t maxElems) {
  const int tag = blockIdx.x;
  pcclTeam team = pcclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  pcclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  const size_t offset = (size_t)(threadIdx.x % slots) * numElems;
  pcclSymPtr<int> dbuf = pcclSymPtr<int>(devWindow, offset * sizeof(int));
  pcclSymPtr<int> sbuf = pcclSymPtr<int>(devWindow, offset * sizeof(int));

  const int activeThreads = (int)blockDim.x;
  const int lastActive = activeThreads - 1;

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int wqesPerOp = 2;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr pcclGinCounter_t counterId = 0;

  for (int i = 0; i < iters; i++) {
    if constexpr (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_CounterInc{counterId}, pcclCoopThread{},
                pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread,
                optFlags | pcclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_CounterInc{counterId}, pcclCoopThread{},
                pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      }
    } else {
      gin.put(team, peer, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_CounterInc{counterId}, pcclCoopThread{},
              pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
    }
    if constexpr (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(pcclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(pcclCoopCta{});
  gin.resetCounter(counterId);
}

template <pcclGinResourceSharingMode rsm>
static void ginPutCounterBWLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                       const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
  const size_t maxElems = args->maxBytes / sizeof(int);
#define LAUNCH_BW_PUT_COUNTER(SKIP, AG) \
  ginPutCounterBwKernel<SKIP, AG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters, numElems, queueDepth, maxElems)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT_COUNTER(true, true);
    else                              LAUNCH_BW_PUT_COUNTER(true, false);
  } else {
    if (args->ginAggregateRequests) LAUNCH_BW_PUT_COUNTER(false, true);
    else                              LAUNCH_BW_PUT_COUNTER(false, false);
  }

#undef LAUNCH_BW_PUT_COUNTER
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutCounterBWLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutCounterBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutCounterBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
