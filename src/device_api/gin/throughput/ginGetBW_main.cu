/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of get. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  ginArgs_t args;
  pcclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = pcclGinRsmGpu;
  caps.allowThreadRsm = false;
  caps.opMask = 0;
  caps.defaultOp = pcclGinOpUnset;
  caps.isSignalOp = false;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  caps.canUseHostMemory = true;
  pcclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = pcclTestGinGetBWLaunch;
  bench.name = "GIN GET BANDWIDTH/MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;

  pcclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
