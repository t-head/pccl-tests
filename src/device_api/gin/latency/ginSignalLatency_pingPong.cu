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

template <bool skipCreditCheck, bool strongSignal, pcclGinResourceSharingMode rsm, bool setSignal = false>
__global__ void ginSignalPingPongKernel(pcclDevComm comm, pcclWindow_t devWindow, int iters, int queueDepth) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  constexpr pcclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    const uint64_t expected = setSignal ? ginSetValue(i) : static_cast<uint64_t>(i) + 1;
    if (comm.rank == 0) {
      gin.waitSignal(pcclCoopThread{}, signalId, expected);
      gin.signal(team, 1, ginSignalAction<setSignal>(signalId, i), pcclCoopThread{}, pcclGin_None{},
                 hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
    } else {
      gin.signal(team, 0, ginSignalAction<setSignal>(signalId, i), pcclCoopThread{}, pcclGin_None{},
                 hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
      gin.waitSignal(pcclCoopThread{}, signalId, expected);
    }
  }
  gin.flush(pcclCoopThread{});
}

template <pcclGinResourceSharingMode rsm, bool setSignal>
static void ginSignalLatencyPingPongLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                               const ginArgs_t* args, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PONG_SIGNAL(SKIP, STRONG) \
  ginSignalPingPongKernel<SKIP, STRONG, rsm, setSignal><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters, queueDepth)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_SIGNAL(true, true);
    else                         LAUNCH_PING_PONG_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_SIGNAL(false, true);
    else                         LAUNCH_PING_PONG_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PONG_SIGNAL
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinSignalLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  pcclTestGinDispatchSignal(args->signalType, [&](auto setSignal) {
    if (args->ginRsm == pcclGinRsmCta) {
      ginSignalLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA, decltype(setSignal)::value>(ctx, stream, args, iters);
    } else {
      ginSignalLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU, decltype(setSignal)::value>(ctx, stream, args, iters);
    }
  });
}
