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

template <bool skipCreditCheck, bool strongSignal, pcclGinResourceSharingMode rsm>
__global__ void ginPutSignalPingKernel(pcclDevComm comm, pcclWindow_t devWindow, size_t numElems, int iters) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> sbuf = pcclSymPtr<int>(devWindow, 0);
  pcclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  constexpr pcclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    /* pccl has a single (strong) signal kind; see the file header. */
    (void)strongSignal;
    gin.put(team, 1, dbuf, sbuf, numElems, pcclGin_SignalInc{signalId}, pcclGin_None{}, pcclCoopThread{},
            pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
    gin.flush(pcclCoopThread{});
  }
}

template <pcclGinResourceSharingMode rsm>
static void ginPutSignalLatencyPingLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                              const ginArgs_t* args, size_t numElems, int iters) {
#define LAUNCH_PING_PUT_SIGNAL(SKIP, STRONG) \
  ginPutSignalPingKernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, numElems, iters)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_PUT_SIGNAL(true, true);
    else                         LAUNCH_PING_PUT_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_PUT_SIGNAL(false, true);
    else                         LAUNCH_PING_PUT_SIGNAL(false, false);
  }

#undef LAUNCH_PING_PUT_SIGNAL
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutSignalLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutSignalLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutSignalLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
