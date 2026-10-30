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
__global__ void ginGetPingKernel(pcclDevComm comm, pcclWindow_t rWindow, pcclWindow_t lWindow, int iters, size_t bytes, bool isRemoteBufDev, bool isLocalBufDev) {
  pcclTeam team = pcclTeamWorld(comm);
  pcclGin gin(comm, 0, rsm);

  pcclSymPtr<int> rbuf = pcclSymPtr<int>(rWindow, 0);
  pcclSymPtr<int> lbuf = pcclSymPtr<int>(lWindow, 0);

  constexpr uint32_t optFlags =
    skipCreditCheck ? pcclGinOptFlagsMaySkipCreditCheck : pcclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    ginGetDispatch(isLocalBufDev, isRemoteBufDev, gin, team, 1, rbuf.window, rbuf.offset, lbuf.window, lbuf.offset, bytes, pcclCoopThread{},
             pcclGin_None{}, optFlags);

    gin.flush(pcclCoopThread{});
  }
}

template <pcclGinResourceSharingMode rsm>
static void ginGetLatencyPingLaunchRsm(const ginContext_t* ctx, hggcStream_t stream,
                                        const ginArgs_t* args, size_t numElems, int iters) {
  pcclWindow_t rWindow = args->remoteMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  pcclWindow_t lWindow = args->localMemoryType == pcclGinMemoryDevice ? ctx->devBufWindow : ctx->hostBufWindow;
  size_t bytes = numElems * sizeof(int);
  bool isRemoteBufDev = args->remoteMemoryType == pcclGinMemoryDevice;
  bool isLocalBufDev = args->localMemoryType == pcclGinMemoryDevice;
#define LAUNCH_PING_GET(SKIP) \
  ginGetPingKernel<SKIP, rsm><<<1, 1, 0, stream>>>(ctx->dcomm, rWindow, lWindow, iters, bytes, isRemoteBufDev, isLocalBufDev)
  if (args->ginSkipCreditCheck) LAUNCH_PING_GET(true);
  else LAUNCH_PING_GET(false);

#undef LAUNCH_PING_GET
  HGGCCHECK_FATAL(hggcGetLastError());
}

void pcclTestGinGetLatencyPingLaunch(const ginContext_t* ctx, hggcStream_t stream, const ginArgs_t* args, size_t numElems, int iters) {
  if (args->ginRsm == pcclGinRsmCta) {
    ginGetLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_CTA>(ctx, stream, args, numElems, iters);
  } else {
    ginGetLatencyPingLaunchRsm<PCCL_GIN_RESOURCE_SHARING_GPU>(ctx, stream, args, numElems, iters);
  }
}
