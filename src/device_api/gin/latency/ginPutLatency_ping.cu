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
__global__ void ginPutPingKernel(pcclDevComm comm, pcclWindow_t dWindow, pcclWindow_t sWindow, size_t numElems, int iters, bool isDestBufDev, bool isSourceBufDev) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> dbuf = pcclSymPtr<int>(dWindow, 0);
  pcclSymPtr<int> sbuf = pcclSymPtr<int>(sWindow, 0);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    ginPutDispatch(isSourceBufDev, isDestBufDev, gin, team, 1, dbuf, sbuf, numElems, pcclGin_None{}, pcclGin_None{}, pcclCoopThread{},
             pcclGin_None{}, hggc::thread_scope_thread, hggc::thread_scope_thread, optFlags);

    gin.flush(pcclCoopThread{});
  }
}

template <pcclGinResourceSharingMode rsm>
static void ginPutLatencyPingLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                        const ginArgs_t* args, size_t numElems, int iters) {
  pcclWindow_t dWindow = args->remoteMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  pcclWindow_t sWindow = args->localMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  bool isDestBufDev = args->remoteMemoryType == pcclGinMemoryDevice;
  bool isSourceBufDev = args->localMemoryType == pcclGinMemoryDevice;
#define LAUNCH_PING_PUT(SKIP) \
  ginPutPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, dWindow, sWindow, numElems, iters, isDestBufDev, isSourceBufDev)

  if (args->ginSkipCreditCheck) LAUNCH_PING_PUT(true);
  else LAUNCH_PING_PUT(false);

#undef LAUNCH_PING_PUT
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinPutLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginPutLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginPutLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
