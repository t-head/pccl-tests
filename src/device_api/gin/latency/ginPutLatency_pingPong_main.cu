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

/*
 * Ping-pong latency of the put family. There is no put+counter variant here: a
 * counter reports local completion, which cannot drive the peer's turn.
 */
int main(int argc, char** argv) {
  ginArgs_t args;
  pcclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = pcclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(pcclGinOpPut) | GIN_OP_BIT(pcclGinOpPutSignal);
  caps.defaultOp = pcclGinOpPut;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  caps.canUseHostMemory = false;
  pcclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;
  switch (args.ginOp) {
  case pcclGinOpPutSignal:
    bench.run = pcclTestGinPutSignalLatencyPingPongLaunch;
    bench.name = "GIN PUT_SIGNAL PING-PONG LATENCY";
    break;
  default:
    bench.run = pcclTestGinPutLatencyPingPongLaunch;
    bench.name = "GIN PUT PING-PONG LATENCY";
    break;
  }

  pcclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
