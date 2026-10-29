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

template <bool skipCreditCheck, pcclGinResourceSharingMode rsm>
__global__ void ginPutCounterPingKernel(pcclDevComm comm, pcclWindow_t devWindow, size_t numElems, int iters,
                                  int queueDepth) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> sbuf = pcclSymPtr<int>(devWindow, 0);
  pcclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  constexpr pcclGinCounter_t counterId = 0;
  uint64_t counterShadow = 0;
  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  for (int i = 0; i < iters; i++) {
    gin.put(team, 1, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_CounterInc{counterId}, pcclCoopThread{},
            pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
    counterShadow++;
    if constexpr (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
    }
    gin.waitCounter(pcclCoopThread{}, counterId, counterShadow);
  }

  gin.flush(pcclCoopThread{});
  gin.resetCounter(counterId);
}

template <pcclGinResourceSharingMode rsm>
static void ginPutCounterLatencyPingLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                               const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PUT_COUNTER(SKIP) \
  ginPutCounterPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, numElems, iters, queueDepth)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PUT_COUNTER(true);
  else LAUNCH_PING_PUT_COUNTER(false);

#undef LAUNCH_PING_PUT_COUNTER
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutCounterLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutCounterLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutCounterLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
