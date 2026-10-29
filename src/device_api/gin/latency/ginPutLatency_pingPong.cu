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
__global__ void ginPutPingPongKernel(pcclDevComm comm, pcclWindow_t devWindow, size_t numElems, int iters,
                                int queueDepth) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> sbuf = pcclSymPtr<int>(devWindow, 0);
  pcclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  int* lastElem = (int*)pcclGetLocalPointer(devWindow, 0) + (numElems - 1);

  int serverSign = 1;
  int clientSign = -1;
  for (int i = 0; i < iters; i++) {
    if (comm.rank == 0) {
      while (*(volatile int*)lastElem != serverSign) continue;
      hggc::atomic_thread_fence(hggc::memory_order_acquire, hggc::thread_scope_system);
      *lastElem = clientSign;
      gin.put(team, 1, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_None{}, pcclCoopThread{},
              pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
    } else {
      *lastElem = serverSign;
      gin.put(team, 0, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_None{}, pcclCoopThread{},
              pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
      if constexpr (skipCreditCheck) {
        if (i % flushEvery == 0) gin.flush(pcclCoopThread{});
      }
      while (*(volatile int*)lastElem != clientSign) continue;
      hggc::atomic_thread_fence(hggc::memory_order_acquire, hggc::thread_scope_system);
    }
    clientSign--;
    serverSign++;
  }
  gin.flush(pcclCoopThread{});
}

template <pcclGinResourceSharingMode rsm>
static void ginPutLatencyPingPongLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                            const ginArgs_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queueDepth;
#define LAUNCH_PING_PONG_PUT(SKIP) \
  ginPutPingPongKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, numElems, iters, queueDepth)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PONG_PUT(true);
  else LAUNCH_PING_PONG_PUT(false);

#undef LAUNCH_PING_PONG_PUT
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutLatencyPingPongLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutLatencyPingPongLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
