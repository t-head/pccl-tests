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
__global__ void ginPutSignalPingPongKernel(pcclDevComm comm, pcclWindow_t devWindow, size_t numElems, int iters,
                                           int queueDepth, int wqesPerOp) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> sbuf = pcclSymPtr<int>(devWindow, 0);
  pcclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int flushEvery = (queueDepth / 2) / wqesPerOp < 1 ? 1 : (queueDepth / 2) / wqesPerOp;

  constexpr pcclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    const uint64_t expected = setSignal ? ginSetValue(i) : static_cast<uint64_t>(i) + 1;
    if (comm.rank == 0) {
      gin.waitSignal(pcclCoopThread{}, signalId, expected);
      gin.put(team, 1, dbuf, sbuf, numElems, ginSignalAction<setSignal>(signalId, i), pcclGin_None{}, pcclCoopThread{},
              pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
    } else {
      gin.put(team, 0, dbuf, sbuf, numElems, ginSignalAction<setSignal>(signalId, i), pcclGin_None{}, pcclCoopThread{},
              pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
      gin.waitSignal(pcclCoopThread{}, signalId, expected);
    }
  }
  gin.flush(pcclCoopThread{});
}

template <pcclGinResourceSharingMode rsm, bool setSignal>
static void ginPutSignalLatencyPingPongLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                                  const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
  const int wqesPerOp = pcclTestGinWqesPerOp(numElems * sizeof(int), pcclGinOpPutSignal);
#define LAUNCH_PING_PONG_PUT_SIGNAL(SKIP, STRONG) \
  ginPutSignalPingPongKernel<SKIP, STRONG, rsm, setSignal><<<1, 1, 0, stream>>>( \
    ctx->dcomm, ctx->devBufWindow, numElems, iters, queueDepth, wqesPerOp)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_PUT_SIGNAL(true, true);
    else                         LAUNCH_PING_PONG_PUT_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PONG_PUT_SIGNAL(false, true);
    else                         LAUNCH_PING_PONG_PUT_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PONG_PUT_SIGNAL
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutSignalLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  pcclTestGinDispatchSignal(args->signalType, [&](auto setSignal) {
    if (args->ginRsm == pcclGinRsmCta) {
      ginPutSignalLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA, decltype(setSignal)::value>(ctx, stream, args, numElems, iters);
    } else {
      ginPutSignalLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU, decltype(setSignal)::value>(ctx, stream, args, numElems, iters);
    }
  });
}
