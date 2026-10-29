/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping-pong latency of signal. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  ginArgs_t args;
  pcclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = pcclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = 0;
  caps.defaultOp = pcclGinOpUnset;
  caps.isSignalOp = true;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  caps.canUseHostMemory = false;
  pcclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = pcclTestGinSignalLatencyPingPongLaunch;
  bench.name = "GIN SIGNAL PING-PONG LATENCY";
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;

  pcclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
