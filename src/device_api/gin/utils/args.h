/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <stddef.h>

typedef enum {
  pcclGinRsmUnset = 0,
  pcclGinRsmThread,
  pcclGinRsmCta,
  pcclGinRsmGpu,
} pcclGinRsm_t;

/*
 * GIN operation under test, selected by --gin_op. A benchmark binary links one
 * test implementation per operation it supports and dispatches to the matching
 * one, so e.g. the put binaries cover plain put, put+signal and put+counter.
 */
typedef enum {
  pcclGinOpUnset = 0,
  pcclGinOpPut,
  pcclGinOpPutSignal,
  pcclGinOpPutCount,
} pcclGinOp_t;

typedef enum {
  pcclGinMemoryDevice = 0,
  pcclGinMemoryHost,
} pcclGinMemoryType_t;

#define GIN_OP_BIT(op) (1u << (unsigned)(op))

typedef struct {
  size_t minBytes;
  size_t maxBytes;
  int stepFactor;
  size_t stepBytes;
  int warmupIters;
  int iters;
  int numCtas;
  bool numCtasSet;
  int numThreads;
  int ginSkipCreditCheck;
  bool ginStrongSignal;
  bool ginAggregateRequests;
  bool ginBidirectional;
  pcclGinMemoryType_t localMemoryType;
  pcclGinMemoryType_t remoteMemoryType;
  int queueDepth;
  pcclGinRsm_t ginRsm;
  pcclGinOp_t ginOp;
} ginArgs_t;

/*
 * What a benchmark binary actually implements. Anything a binary does not
 * declare here is rejected by pcclTestGinConfigureArgs with a diagnostic
 * rather than being silently ignored.
 *
 *   defaultRsm / allowThreadRsm – --gin_rsm default (gpu) and whether thread
 *                                 mode is usable (latency yes, throughput no);
 *                                 with pccl thread mode is additionally
 *                                 rejected for every binary.
 *   opMask                      – OR of GIN_OP_BIT() values --gin_op accepts;
 *                                 0 for binaries with a single operation
 *                                 (the get and signal tests).
 *   defaultOp                   – used when --gin_op is omitted.
 *   isSignalOp                  – the binary's operation is itself a signal,
 *                                 so --gin_strong_signal applies regardless
 *                                 of --gin_op.
 *   allowAggregateRequests      – --gin_ag (throughput only).
 *   allowBidirectional          – --gin_bd (throughput only).
 *   allowMultiCtaThreads        – -c/-t other than 1 (throughput only;
 *                                 latency requires a single CTA and thread).
 *   canUseHostMemory            – --local_mem_type/--remote_mem_type options for CPU-backed memory support.
 */
typedef struct {
  pcclGinRsm_t defaultRsm;
  bool allowThreadRsm;
  unsigned opMask;
  pcclGinOp_t defaultOp;
  bool isSignalOp;
  bool allowAggregateRequests;
  bool allowBidirectional;
  bool allowMultiCtaThreads;
  bool canUseHostMemory;
} ginTestCaps_t;

void pcclTestGinParseArgs(int argc, char** argv, ginArgs_t* args);
void pcclTestGinConfigureArgs(ginArgs_t* args, const ginTestCaps_t* caps);
const char* pcclTestGinOpName(pcclGinOp_t op);
const char* pcclTestGinRsmName(pcclGinRsm_t rsm);
const char* pcclTestGinMemoryTypeName(pcclGinMemoryType_t memory);
