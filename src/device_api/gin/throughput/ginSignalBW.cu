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

template <bool skipCreditCheck, bool aggregateRequests, bool strongSignal,
          pcclGinResourceSharingMode rsm>
__global__ void ginSignalBwKernel(pcclDevComm comm, pcclWindow_t devWindow, int iters, int queueDepth) {
  const int tag = blockIdx.x;
  pcclTeam team = pcclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  pcclGin gin(comm, tag, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr pcclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    /* pccl has a single (strong) signal kind; see the file header. */
    (void)strongSignal;
    if constexpr (aggregateRequests) {
      if (threadIdx.x < blockDim.x - 1) {
        gin.signal(team, peer, pcclGin_SignalInc{signalId}, pcclCoopThread{}, pcclGin_None{},
                   hggc::thread_scope_thread, hggc::thread_scope_thread,
                   optFlags | pcclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == blockDim.x - 1) {
        gin.signal(team, peer, pcclGin_SignalInc{signalId}, pcclCoopThread{}, pcclGin_None{},
                   hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      }
    } else {
      gin.signal(team, peer, pcclGin_SignalInc{signalId}, pcclCoopThread{}, pcclGin_None{},
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
static void ginSignalBWLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                  const ginArgs_t* args, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_BW_SIGNAL(SKIP, AG, STRONG) \
  ginSignalBwKernel<SKIP, AG, STRONG, rsm><<<args->numCtas, args->numThreads, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters, queueDepth)

  if (args->ginSkipCreditCheck) {
    if (args->ginAggregateRequests) {
      if (args->ginStrongSignal) LAUNCH_BW_SIGNAL(true, true, true);
      else                         LAUNCH_BW_SIGNAL(true, true, false);
    } else {
      if (args->ginStrongSignal) LAUNCH_BW_SIGNAL(true, false, true);
      else                         LAUNCH_BW_SIGNAL(true, false, false);
    }
  } else {
    if (args->ginAggregateRequests) {
      if (args->ginStrongSignal) LAUNCH_BW_SIGNAL(false, true, true);
      else                         LAUNCH_BW_SIGNAL(false, true, false);
    } else {
      if (args->ginStrongSignal) LAUNCH_BW_SIGNAL(false, false, true);
      else                         LAUNCH_BW_SIGNAL(false, false, false);
    }
  }

#undef LAUNCH_BW_SIGNAL
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinSignalBWLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginSignalBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, iters);
  } else {
    ginSignalBWLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, iters);
  }
}
