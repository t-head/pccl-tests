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

/*
 * Message rate of signal. Signals carry no payload, so bandwidth is reported as
 * zero and only the message rate column is meaningful.
 */
int main(int argc, char** argv) {
  ginArgs_t args;
  pcclTestGinParseArgs(argc, argv, &args);

  ginTestCaps_t caps = {};
  caps.defaultRsm = pcclGinRsmGpu;
  caps.allowThreadRsm = false;
  caps.opMask = 0;
  caps.defaultOp = pcclGinOpUnset;
  caps.isSignalOp = true;
  caps.allowAggregateRequests = true;
  caps.allowBidirectional = true;
  caps.allowMultiCtaThreads = true;
  caps.canUseHostMemory = false;
  pcclTestGinConfigureArgs(&args, &caps);

  ginBenchmark_t bench = {};
  bench.run = pcclTestGinSignalBWLaunch;
  bench.name = "GIN SIGNAL MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  bench.payloadMode = GIN_THROUGHPUT_PAYLOAD_NONE;

  pcclTestGinPerfRun(argc, argv, &args, &bench);
  return 0;
}
