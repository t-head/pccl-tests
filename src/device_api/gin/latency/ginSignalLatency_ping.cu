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
__global__ void ginSignalPingKernel(pcclDevComm comm, pcclWindow_t devWindow, int iters) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  constexpr pcclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    /* pccl has a single (strong) signal kind; see the file header. */
    (void)strongSignal;
    gin.signal(team, 1, pcclGin_SignalInc{signalId}, pcclCoopThread{}, pcclGin_None{},
               hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);
    gin.flush(pcclCoopThread{});
  }
}

template <pcclGinResourceSharingMode rsm>
static void ginSignalLatencyPingLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                           const ginArgs_t* args, int iters) {
#define LAUNCH_PING_SIGNAL(SKIP, STRONG) \
  ginSignalPingKernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, ctx->devBufWindow, iters)

  if (args->ginSkipCreditCheck) {
    if (args->ginStrongSignal) LAUNCH_PING_SIGNAL(true, true);
    else                         LAUNCH_PING_SIGNAL(true, false);
  } else {
    if (args->ginStrongSignal) LAUNCH_PING_SIGNAL(false, true);
    else                         LAUNCH_PING_SIGNAL(false, false);
  }

#undef LAUNCH_PING_SIGNAL
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinSignalLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginSignalLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, iters);
  } else {
    ginSignalLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, iters);
  }
}
