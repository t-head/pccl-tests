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

/* Ping latency of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  ginArgs_t args;
  pcclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = pcclGinRsmGpu;
  caps.allowThreadRsm = true;
  caps.opMask = GIN_OP_BIT(pcclGinOpPut) | GIN_OP_BIT(pcclGinOpPutSignal) | GIN_OP_BIT(pcclGinOpPutCount);
  caps.defaultOp = pcclGinOpPut;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = false;
  caps.allowBidirectional = false;
  caps.allowMultiCtaThreads = false;
  caps.canUseHostMemory = true;
  pcclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING;
  switch (args.ginOp) {
  case pcclGinOpPutSignal:
    if (args.localMemoryType == pcclGinMemoryHost || args.remoteMemoryType == pcclGinMemoryHost) {
      fprintf(stderr, "Error: host memory type is not supported in GIN PUT signal\n");
      exit(EXIT_FAILURE);
    }
    bench.run = pcclTestGinPutSignalLatencyPingLaunch;
    bench.name = "GIN PUT_SIGNAL PING LATENCY";
    break;
  case pcclGinOpPutCount:
    if (args.localMemoryType == pcclGinMemoryHost || args.remoteMemoryType == pcclGinMemoryHost) {
      fprintf(stderr, "Error: host memory type is not supported in GIN PUT counter\n");
      exit(EXIT_FAILURE);
    }
    bench.run = pcclTestGinPutCounterLatencyPingLaunch;
    bench.name = "GIN PUT_COUNTER PING LATENCY";
    break;
  default:
    bench.run = pcclTestGinPutLatencyPingLaunch;
    bench.name = "GIN PUT PING LATENCY";
    break;
  }

  pcclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
